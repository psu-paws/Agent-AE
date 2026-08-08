from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class KVHandoffCompleteEvent(BaseEvent):
    """Fire at (decode_admission + kv_transfer_time) to start decoding a request
    whose KV has finished arriving.

    The decode replica's gate has already reserved blocks for the request, which
    is what let the pull begin; until this event it sits in `_transferring`,
    holding that reservation but ineligible for batching.
    """

    def __init__(
        self,
        time: float,
        request: Request,
        decode_replica_id: ReplicaId,
    ):
        super().__init__(time, EventType.KV_HANDOFF_COMPLETE)
        self._request = request
        self._decode_replica_id = decode_replica_id

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        decode_scheduler = global_scheduler.get_replica_scheduler(
            self._decode_replica_id
        )
        decode_scheduler.start_decode(self._request)
        return [ReplicaScheduleEvent(self.time, self._decode_replica_id)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "request_id": self._request.id,
            "decode_replica_id": str(self._decode_replica_id),
            "kv_transfer_tokens": self._request._kv_transfer_tokens,
        }
