from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        logger.debug(f"Request: {self._request.id} arrived at {self.time}")
        global_scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self._request)
        return [GlobalScheduleEvent(self.time)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "request": self._request.id,
        }
