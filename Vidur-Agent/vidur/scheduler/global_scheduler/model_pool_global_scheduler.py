from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from vidur.entities import Request
from vidur.kv_cache.utils import hash_request_tokens
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId

logger = init_logger(__name__)


class ModelPoolGlobalScheduler(BaseGlobalScheduler):
    """Routes each request to the replica pool serving its model.

    The trace fixes which model a request targets (``model_id`` column);
    this scheduler chooses which replica within that model's pool should serve.
    Choosing the replica is decision between KV-aware and load-aware routing.
    It is delegated to ``_pick_prefill_replica`` / ``_pick_decode_replica``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replica_id_list = sorted(self._replicas.keys())
        cluster_config = self._config.cluster_config

        # model_id -> replica IDs serving that model in each role.
        self._prefill_pool: Dict[int, List[ReplicaId]] = {}
        self._decode_pool: Dict[int, List[ReplicaId]] = {}
        self._prefill_replica_to_model: Dict[ReplicaId, int] = {}

        # Two cluster shapes. With `replica_groups_pools`, replicas are split into
        # prefill and decode roles and a request is handed off between them.
        # Without it every replica is aggregated: a model's pool is simply its
        # replica group, so model_id indexes the groups in declaration order and
        # the same routing policy applies -- there is no separate degenerate path.
        self._is_pd = bool(cluster_config._pd_pools)

        if self._is_pd:
            for pool_def in cluster_config._pd_pools:
                model_id = pool_def["model_id"]
                for p_idx in pool_def["prefill"]:
                    p_replica_id = self._replica_id_list[p_idx]
                    self._prefill_pool.setdefault(model_id, []).append(p_replica_id)
                    self._prefill_replica_to_model[p_replica_id] = model_id
                for d_idx in pool_def["decode"]:
                    d_replica_id = self._replica_id_list[d_idx]
                    self._decode_pool.setdefault(model_id, []).append(d_replica_id)
        else:
            for i, replica_id in enumerate(self._replica_id_list):
                group_id = cluster_config.get_replica_group(i)
                self._prefill_pool.setdefault(group_id, []).append(replica_id)

        # Per-model staging buffers: requests awaiting a prefill replica.
        self._pending: Dict[int, List[Request]] = {
            model_id: [] for model_id in self._prefill_pool
        }
        self._cross_node_pairs: Dict[tuple, int] = self._build_cross_node_pairs()
        logger.info("Prefill pools: %s", {m: [r.id for r in rs]
                                          for m, rs in self._prefill_pool.items()})
        logger.info("Decode pools: %s", {m: [r.id for r in rs]
                                         for m, rs in self._decode_pool.items()})
        logger.info("Cross-node pairs: %s", self._cross_node_pairs)

    @abstractmethod
    def _pick_prefill_replica(self, model_id: int, request: Request) -> ReplicaId:
        """Choose which replica serving this model handles the request."""

    @abstractmethod
    def _pick_decode_replica(
        self, model_id: int, request: Request) -> Optional[ReplicaId]:
        """Choose the decode replica, or None when there is no decode pool."""

    def _begin_tick(self) -> None:
        """Hook for per-tick setup a policy may need. No-OP by default."""


    def _build_cross_node_pairs(self) -> Dict[tuple, int]:
        """Return {(prefill_rid, decode_rid): decode_tp_size} for cross-node pairs.
        Reads the `cross_node` flag from the pool config. """
        cluster_config = self._config.cluster_config
        cross_node: Dict[tuple, int] = {}
        for pool in cluster_config._pd_pools:
            if not pool.get("cross_node", False):
                continue
            for p_idx in pool["prefill"]:
                for d_idx in pool["decode"]:
                    p_replica_id = self._replica_id_list[p_idx]
                    d_replica_id = self._replica_id_list[d_idx]
                    decode_tp = cluster_config._replica_id_to_config[
                        d_idx
                    ].tensor_parallel_size
                    cross_node[(p_replica_id, d_replica_id)] = decode_tp
        return cross_node

    def _model_key(self, request: Request) -> int:
        """Which pool serves this request.

        A trace without a `model_id` column is only unambiguous when the cluster
        serves a single model; otherwise there is no way to know which pool the
        request belongs to, and that is a configuration error worth failing on.
        """
        model_id = request._model_id
        if model_id is None:
            assert len(self._prefill_pool) == 1, (
                f"Trace has no model_id column, but the cluster declares "
                f"{len(self._prefill_pool)} model pools ({sorted(self._prefill_pool)}). "
                f"Add a model_id column, or configure a single model."
            )
            return next(iter(self._prefill_pool))
        assert model_id in self._prefill_pool, (
            f"Trace requests model_id {model_id}, but pools exist only for "
            f"{sorted(self._prefill_pool)}. Every model in the trace needs a "
            f"replica group (aggregated) or a replica_groups_pools entry (PD)."
        )
        return model_id

    def add_request(self, request: Request) -> None:
        """Route an incoming request into its model's staging buffer."""
        self._slo_manager.set_slos(request)
        self._pending[self._model_key(request)].append(request)

    def is_empty(self) -> bool:
        return (
            all(len(buf) == 0 for buf in self._pending.values())
            and len(self._request_queue) == 0
            and all(s.is_empty() for s in self._replica_schedulers.values())
        )

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        """Drain the staging buffers and assign each request to a replica."""
        self._begin_tick()
        request_mapping = []

        for model_id, buf in self._pending.items():
            if not buf:
                continue
            buf.sort(key=lambda r: (r.arrived_at, r.id))
            while buf:
                request = buf.pop(0)
                replica_id = self._pick_prefill_replica(model_id, request)
                if self._is_pd:
                    # Assign the decode replica at dispatch time, so prefill
                    # blocks can stream to it while prefill is still running.
                    request._assigned_decode_replica_id = self._pick_decode_replica(
                        model_id, request
                    )
                request_mapping.append((replica_id, request))

        assert not self._request_queue, (
            f"Unexpected requests in global queue: "
            f"{[r.id for r in self._request_queue]}"
        )
        return request_mapping

    def _compute_kv_stall_delay_s(
        self, predictor, decode_tp: int, delta_tokens: int
    ) -> float:
        """KV-transfer stall in seconds, as a sequential transfer of delta_tokens.
        Blocks streamed during prefill are assumed hidden by computation.
        """
        if delta_tokens <= 0:
            return 0.0
        if decode_tp > 0:
            stall_ms = predictor.get_kv_transfer_time_cross_node(delta_tokens, decode_tp)
        else:
            stall_ms = predictor.get_kv_transfer_time(delta_tokens)
        return stall_ms / 1e3

    def on_prefill_end(self, request) -> None:
        """Hand a prefill-complete request off to its decode replica's queue."""
        prefill_replica_id = request.replica_id

        # Zero-decode requests finish.
        if request.completed:
            return

        # Aggregated replicas - no handoff.
        model_id = self._prefill_replica_to_model.get(prefill_replica_id)
        if model_id is None:
            return

        decode_replica_id = request._assigned_decode_replica_id
        if decode_replica_id is None:
            return

        # Prefill releases its KV at handoff
        self.get_replica_scheduler(prefill_replica_id).remove_prefill_end_request(
            request
        )
        request.on_pd_transfer(request.prefill_completed_at, decode_replica_id)
        self.get_replica_scheduler(decode_replica_id).add_decode_request(request)

    def on_decode_admit(self, request) -> float:
        """Price the KV pull at the moment the decode replica admits the request.

        Blocks streamed to decode during prefill that are still resident cost
        nothing. but becomes the whole request when decode evicted those blocks in stressed setup.
        """
        prefill_replica_id = request.prefill_replica_id
        decode_replica_id = request.decode_replica_id
        decode_kv_mgr = self.get_replica_scheduler(decode_replica_id)._kv_cache_manager

        # Only from block 0 is reusable
        decode_cached_blocks = 0
        if decode_kv_mgr.enable_caching:
            block_hashes = decode_kv_mgr.req_to_block_hashes.get(request.id)
            if not block_hashes:
                block_hashes = hash_request_tokens(
                    decode_kv_mgr.caching_hash_fn, decode_kv_mgr.block_size, request
                )
            for h in block_hashes:
                if h not in decode_kv_mgr.block_pool.cached_block_hash_to_block:
                    break
                decode_cached_blocks += 1

        block_size = decode_kv_mgr.block_size
        delta_tokens = request.num_prefill_tokens - decode_cached_blocks * block_size
        request._kv_transfer_tokens = max(0, delta_tokens)

        predictor = self.get_execution_time_predictor(prefill_replica_id)
        decode_tp = self._cross_node_pairs.get(
            (prefill_replica_id, decode_replica_id), 0
        )
        return self._compute_kv_stall_delay_s(
            predictor, decode_tp, request._kv_transfer_tokens
        )
