from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.execution_time import ExecutionTime
from vidur.entities.request import Request
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class BatchStage(BaseEntity):
    def __init__(
        self,
        batch_id: int,
        replica_id: ReplicaId,
        stage_id: int,
        execution_time: ExecutionTime,
        requests: List[Request],
        num_tokens: List[Request],
    ) -> None:
        self._id = BatchStage.generate_id()

        self._requests = requests
        self._num_tokens = num_tokens
        self._batch_id = batch_id
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._execution_time = execution_time

        self._total_execution_time = self._execution_time.total_time
        self._model_execution_time = self._execution_time.model_time

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def execution_time(self) -> float:
        return self._total_execution_time

    @property
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    def replica_id(self) -> ReplicaId:
        return self._replica_id

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def size(self) -> int:
        return len(self._requests)

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_stage_schedule(time)

    def on_stage_end(
        self,
        time: float,
    ) -> None:
        assert (
            time == self._scheduled_at + self._total_execution_time
        ), f"{time} != {self._scheduled_at} + {self._total_execution_time}"

        self._completed_at = time

        for request in self._requests:
            request.on_batch_stage_end(
                time, self._total_execution_time, self._model_execution_time
            )

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "execution_time": self._execution_time.to_dict(),
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "replica_id": self._replica_id,
            "batch_id": self._batch_id,
            "stage_id": self._stage_id,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
        }

    def to_chrome_trace(self, time: int) -> dict:
        return {
            "name": f"{self.request_ids}",
            "ph": "X",
            "ts": (time - self._total_execution_time) * 1e6,
            "dur": self._total_execution_time * 1e6,
            "pid": str(self._replica_id),
            "tid": self._stage_id,
            "args": {
                "batch_id": self._batch_id,
                "batch_size": self.size,
                "request_ids": self.request_ids,
                "num_tokens": self._num_tokens,
                "execution_time": self._execution_time.to_dict(),
                "requests": [request.to_dict() for request in self._requests],
            },
        }
