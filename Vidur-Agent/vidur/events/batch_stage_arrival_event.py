from typing import List

from vidur.entities.batch import Batch
from vidur.events import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class BatchStageArrivalEvent(BaseEvent):
    def __init__(self, time: float, replica_id: ReplicaId, stage_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

        global_scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        ).add_batch(self._batch)

        return [
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
            )
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
        }
