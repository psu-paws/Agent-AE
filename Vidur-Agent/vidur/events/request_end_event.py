from typing import List

from vidur.entities import Request
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RequestEndEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_END)

        self._request = request

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        logger.debug(f"Request: {self._request.id} ended at {self.time}")
        global_scheduler.on_request_end(self._request)
        metrics_store.on_request_end(self._request)
        return []

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "request": self._request.id,
        }
