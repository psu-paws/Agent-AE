from typing import List, Optional

from vidur.entities.batch import Batch
from vidur.entities.request import Request


class ReplicaSchedulerOutput:
    def __init__(self, batch: Optional[Batch], requeued_requests: List[Request]):
        self._batch = batch
        self._requeued_requests = requeued_requests

    @property
    def batch(self) -> Optional[Batch]:
        return self._batch

    @property
    def requeued_requests(self) -> List[Request]:
        return self._requeued_requests
