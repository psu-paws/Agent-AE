import random
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


class BaseLOPGlobalScheduler(BaseGlobalScheduler):
    """
    Base Least outstanding prefills (LOP) early binding global scheduler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We keep track of the number of pending prefill tokens for each replica
        # and use this to find the replica with the least number of pending prefill tokens
        self._pending_prefills_map = {
            replica_scheduler.replica_id: 0
            for replica_scheduler in self._replica_schedulers.values()
        }

    def _increment_pending_prefills(
        self, replica_id: ReplicaId, request: Request
    ) -> None:
        pass

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            min_pending_prefills = min(self._pending_prefills_map.values())
            replica_id = self._random_number_generator.choice(
                [
                    replica_id
                    for replica_id, pending_prefills in self._pending_prefills_map.items()
                    if pending_prefills == min_pending_prefills
                ]
            )
            self._increment_pending_prefills(replica_id, request)
            request_mapping.append((replica_id, request))

        return request_mapping
