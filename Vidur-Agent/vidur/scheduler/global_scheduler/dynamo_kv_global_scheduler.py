from typing import List, Optional

from vidur.entities import Request
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.model_pool_global_scheduler import (
    ModelPoolGlobalScheduler,
)
from vidur.types.replica_id import ReplicaId

logger = init_logger(__name__)


class DynamoStyleKVGlobalScheduler(ModelPoolGlobalScheduler):
    """Inspired by Dynamo's KV-router scoring.
    (https://github.com/ai-dynamo/dynamo/blob/0dd5374474c497f3dacc4fabdbac5c6f275e5acb/lib/kv-router/src/scheduling/selector.rs#L138)
    Everything structural is inherited from ModelPoolGlobalScheduler,. Only the replica scoring is
    supplied here.
        logit = overlap_score_weight * potential_prefill_block + decode_block
    ``potential_prefill_block = uncached prefill_token / block_size``
    ``decode_block`` is the worker's current decode load
    """

    def _cost(self, replica_id: ReplicaId, request: Request) -> float:
        config = self._config.cluster_config.global_scheduler_config
        replica_scheduler = self.get_replica_scheduler(replica_id)
        kv_cache_manager = replica_scheduler._kv_cache_manager
        block_size = kv_cache_manager.block_size

        prefill_token = max(
            0,
            request.num_prefill_tokens
            - replica_scheduler.get_cached_prefill_length(request),
        )
        potential_prefill_block = prefill_token / block_size

        block_pool = kv_cache_manager.block_pool
        decode_block = block_pool.num_gpu_blocks - block_pool.get_num_free_blocks()

        return config.overlap_score_weight * potential_prefill_block + decode_block

    def _argmin_dynamo(self, pool: List[ReplicaId], request: Request) -> ReplicaId:
        best_replica_id = None
        best_cost = float("inf")
        for replica_id in pool:
            cost = self._cost(replica_id, request)
            if cost < best_cost:
                best_cost = cost
                best_replica_id = replica_id
        return best_replica_id

    def _pick_prefill_replica(self, model_id: int, request: Request) -> ReplicaId:
        pool = self._prefill_pool[model_id]
        if len(pool) == 1:
            return pool[0]
        return self._argmin_dynamo(pool, request)

    def _pick_decode_replica(
        self, model_id: int, request: Request
    ) -> Optional[ReplicaId]:
        pool = self._decode_pool.get(model_id)
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        return self._argmin_dynamo(pool, request)
