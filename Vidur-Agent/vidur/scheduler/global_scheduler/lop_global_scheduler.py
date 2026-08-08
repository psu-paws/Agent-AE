from vidur.entities.request import Request
from vidur.scheduler.global_scheduler.base_lop_global_scheduler import (
    BaseLOPGlobalScheduler,
)


class LOPGlobalScheduler(BaseLOPGlobalScheduler):
    def _increment_pending_prefills(self, replica_id: int, request: Request) -> None:
        self._pending_prefills_map[replica_id] += request.num_prefill_tokens

    def on_prefill_end(self, request):
        self._pending_prefills_map[request.replica_id] -= request.num_prefill_tokens
