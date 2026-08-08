from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


class StickyRoundRobinGlobalScheduler(BaseGlobalScheduler):
    """
    Sticky Prefix cache aware early binding global scheduler.
    Note: For a new `session`, use round robin policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._session_counter = 0
        self._replica_id_list = sorted(self._replicas.keys())
        self._session_to_replica_map = {}

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            assert (
                request.session_id is not None
            ), "`session_id` is required for sticky_prefix_cache_aware global scheduler"
            # If the session is new, assign a replica using round robin policy
            if request.session_id not in self._session_to_replica_map:
                replica_id = self._replica_id_list[
                    self._session_counter % self._num_replicas
                ]
                self._session_counter += 1
                self._session_to_replica_map[request.session_id] = replica_id

            replica_id = self._session_to_replica_map[request.session_id]
            request_mapping.append((replica_id, request))

        return request_mapping
