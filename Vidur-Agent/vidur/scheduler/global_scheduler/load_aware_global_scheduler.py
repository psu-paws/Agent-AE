from collections import defaultdict
from typing import Dict, Optional

from vidur.entities import Request
from vidur.scheduler.global_scheduler.model_pool_global_scheduler import (
    ModelPoolGlobalScheduler,
)
from vidur.types.replica_id import ReplicaId


class LoadAwareGlobalScheduler(ModelPoolGlobalScheduler):
    """Experimental KV-affinity / Load aware routing that scores running/queued requests.

    Same shape as DynamoStyleKVGlobalScheduler, but every term is measured
    in *remaining prefill tokens*, so work not yet started still counts:

        score = running_tokens   # in-flight prefill on that replica
              + waiting_tokens   # queued there, each minus its own cache hit
              + tick_tokens      # dispatched earlier in this same tick
              + this_request     # its prefill, minus its cache hit here

    ``tick_tokens`` is aggressive term: it counts requests
    dispatched earlier in this same scheduling tick. Without it a burst of
    simultaneous arrivals all score against identical state.

    Decode replicas are chosen by longest cached prefix alone.
    With ``use_load_aware_routing=False``, or a pool of one, prefill selection
    degrades to round-robin.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tokens dispatched this tick but not yet in any replica queue.
        self._tick_assigned_tokens: Dict[ReplicaId, int] = defaultdict(int)
        # Running + waiting prefill load per replica, refreshed each tick.
        self._tick_base_load: Dict[ReplicaId, int] = {}
        # get_cached_prefill_length per (request, replica) for this tick.
        self._tick_hit_cache: Dict[tuple, int] = {}
        # Round-robin cursor per model, used when scoring is disabled.
        self._prefill_rr: Dict[int, int] = {m: 0 for m in self._prefill_pool}

    def _begin_tick(self) -> None:
        self._tick_assigned_tokens.clear()
        self._tick_hit_cache.clear()
        if (
            self._prefill_pool
            and self._config.cluster_config.global_scheduler_config.use_load_aware_routing
            and any(len(p) > 1 for p in self._prefill_pool.values())
        ):
            self._build_tick_base_load()

    def _build_tick_base_load(self) -> None:
        """Snapshot running + waiting prefill load per replica, once per tick."""
        for pool in self._prefill_pool.values():
            for replica_id in pool:
                sched = self.get_replica_scheduler(replica_id)
                running_load = sum(
                    r.num_prefill_tokens - r.num_processed_tokens
                    for r in sched._running
                    if not r.is_prefill_complete
                )
                waiting_load = sum(
                    r.num_prefill_tokens - sched.get_cached_prefill_length(r)
                    for r in sched._waiting_queue.to_list()
                )
                self._tick_base_load[replica_id] = running_load + waiting_load

    def _pick_prefill_replica(self, model_id: int, request: Request) -> ReplicaId:
        pool = self._prefill_pool[model_id]
        use_scoring = (
            self._config.cluster_config.global_scheduler_config.use_load_aware_routing
            and len(pool) > 1
        )
        if not use_scoring:
            idx = self._prefill_rr[model_id] % len(pool)
            self._prefill_rr[model_id] = idx + 1
            return pool[idx]

        best_replica_id = None
        best_score = float("inf")
        best_hit = 0
        for replica_id in pool:
            key = (request.id, replica_id)
            if key not in self._tick_hit_cache:
                self._tick_hit_cache[key] = self.get_replica_scheduler(
                    replica_id
                ).get_cached_prefill_length(request)
            hit_tokens = self._tick_hit_cache[key]
            effective = request.num_prefill_tokens - hit_tokens
            score = (
                self._tick_base_load.get(replica_id, 0)
                + self._tick_assigned_tokens[replica_id]
                + effective
            )
            if score < best_score:
                best_score = score
                best_replica_id = replica_id
                best_hit = hit_tokens

        self._tick_assigned_tokens[best_replica_id] += (
            request.num_prefill_tokens - best_hit
        )
        return best_replica_id

    def _pick_decode_replica(
        self, model_id: int, request: Request
    ) -> Optional[ReplicaId]:
        pool = self._decode_pool.get(model_id)
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]

        best_replica_id = None
        best_hit = -1
        for replica_id in pool:
            hit = self.get_replica_scheduler(replica_id).get_cached_prefill_length(
                request
            )
            if hit > best_hit:
                best_hit = hit
                best_replica_id = replica_id
        return best_replica_id
