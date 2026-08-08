import random
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        self._pending_requests_map = {
            replica_scheduler.replica_id: 0
            for replica_scheduler in self._replica_schedulers.values()
        }

    def on_request_end(self, request: Request) -> None:
        replica_id = request.replica_id
        self._pending_requests_map[replica_id] -= 1

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            min_pending_requests = min(self._pending_requests_map.values())
            replica_id = self._random_number_generator.choice(
                [
                    replica_id
                    for replica_id, pending_requests in self._pending_requests_map.items()
                    if pending_requests == min_pending_requests
                ]
            )
            self._pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))

        return request_mapping
