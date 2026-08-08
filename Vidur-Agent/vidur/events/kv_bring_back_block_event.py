from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class KVBringBackBlockEvent(BaseEvent):
    """Fire at (batch_end + transfer_time) to register a decode KV block on
    the prefill replica's prefix cache, enabling future requests with the same
    prefix to reuse it without recomputation."""

    def __init__(
        self,
        time: float,
        request: Request,
        prefill_replica_id: ReplicaId,
        block_index: int,
    ):
        super().__init__(time, EventType.KV_BRING_BACK_BLOCK)
        self._request = request
        self._prefill_replica_id = prefill_replica_id
        self._block_index = block_index

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        prefill_scheduler = global_scheduler.get_replica_scheduler(
            self._prefill_replica_id
        )
        prefill_scheduler._kv_cache_manager.register_brought_back_block(
            self._request, self._block_index
        )
        return []

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "request_id": self._request.id,
            "prefill_replica_id": str(self._prefill_replica_id),
            "block_index": self._block_index,
        }
