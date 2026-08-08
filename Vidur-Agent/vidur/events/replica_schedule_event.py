from typing import List

from vidur.entities.batch import Batch
from vidur.events import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: ReplicaId):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id

        self._batches: List[Batch] = []

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        # Attempt to create batches
        replica_scheduler = global_scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler_outputs: List[ReplicaSchedulerOutput] = []
        while replica_scheduler.can_schedule():
            replica_scheduler_output = replica_scheduler.on_schedule(self.time)
            replica_scheduler_outputs.append(replica_scheduler_output)
            if not replica_scheduler_output.batch:
                break

        # Process the batches formed, schedule next events and update metrics
        self._batches = [
            replica_scheduler_output.batch
            for replica_scheduler_output in replica_scheduler_outputs
            if replica_scheduler_output.batch
        ]
        self._requeued_requests = set(
            request
            for replica_scheduler_output in replica_scheduler_outputs
            for request in replica_scheduler_output.requeued_requests
        )

        for batch in self._batches:
            batch.on_schedule(self.time)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, self._batches, memory_usage_percent
        )

        next_events = []
        pop_admitted = getattr(replica_scheduler, "pop_newly_admitted_decode", None)
        if pop_admitted is not None:
            from vidur.events.kv_handoff_complete_event import KVHandoffCompleteEvent

            for request in pop_admitted():
                delay_s = global_scheduler.on_decode_admit(request)
                next_events.append(
                    KVHandoffCompleteEvent(
                        self.time + delay_s, request, self._replica_id
                    )
                )
        if self._requeued_requests:
            next_events.append(ReplicaScheduleEvent(self.time, self._replica_id))

        return next_events + [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "replica_id": self._replica_id,
            "batch_ids": [batch.id for batch in self._batches],
            "requeued_request_ids": [request.id for request in self._requeued_requests],
        }
