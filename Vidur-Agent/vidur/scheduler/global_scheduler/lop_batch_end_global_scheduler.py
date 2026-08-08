from vidur.entities.batch import Batch
from vidur.entities.request import Request
from vidur.scheduler.global_scheduler.base_lop_global_scheduler import (
    BaseLOPGlobalScheduler,
)
from vidur.types.replica_id import ReplicaId


class LOPBatchEndGlobalScheduler(BaseLOPGlobalScheduler):
    def _increment_pending_prefills(
        self, replica_id: ReplicaId, request: Request
    ) -> None:
        self._pending_prefills_map[replica_id] += request.num_prefill_tokens

    def on_batch_end(self, batch: Batch) -> None:
        self._pending_prefills_map[batch.replica_id] -= batch.num_prefill_tokens
