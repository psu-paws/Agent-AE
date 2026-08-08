from vidur.entities.request import Request
from vidur.scheduler.global_scheduler.base_lop_global_scheduler import (
    BaseLOPGlobalScheduler,
)
from vidur.types.replica_id import ReplicaId


class LOPBinaryGlobalScheduler(BaseLOPGlobalScheduler):
    def _increment_pending_prefills(
        self, replica_id: ReplicaId, request: Request
    ) -> None:
        self._pending_prefills_map[replica_id] += 1

    def on_prefill_end(self, request):
        self._pending_prefills_map[request.replica_id] -= 1
