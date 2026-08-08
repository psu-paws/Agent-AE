from typing import List

from vidur.events import BaseEvent
from vidur.metrics.cluster_metrics_store import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType


class GlobalScheduleEvent(BaseEvent):
    def __init__(self, time: float):
        super().__init__(time, EventType.GLOBAL_SCHEDULE)

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        self._replica_set = set()
        self._request_mapping = scheduler.schedule()

        for replica_id, request in self._request_mapping:
            self._replica_set.add(replica_id)
            scheduler.get_replica_scheduler(replica_id).add_request(request)

        return [
            ReplicaScheduleEvent(self.time, replica_id)
            for replica_id in self._replica_set
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "replica_set": self._replica_set,
            "request_mapping": [
                (replica_id, request.id)
                for replica_id, request in self._request_mapping
            ],
        }
