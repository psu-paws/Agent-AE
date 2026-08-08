from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


class LOPUncachedGlobalScheduler(BaseGlobalScheduler):
    """
    Least Outstandings Prefill tokens Prefix cache aware early binding global scheduler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._pending_prefills_map = {
            replica_scheduler.replica_id: 0
            for replica_scheduler in self._replica_schedulers.values()
        }
        self._cached_prefill_length_map = {}

    def _get_num_prefill_tokens_uncached(
        self, request: Request, replica_id: ReplicaId
    ) -> int:
        key = (request.id, replica_id)
        if key not in self._cached_prefill_length_map:
            cached_prefill_length = self._replica_schedulers[
                replica_id
            ].get_cached_prefill_length(request)
            self._cached_prefill_length_map[key] = cached_prefill_length
        return request.num_prefill_tokens - self._cached_prefill_length_map[key]

    def _increment_pending_prefills(
        self, replica_id: ReplicaId, request: Request
    ) -> None:
        self._pending_prefills_map[replica_id] += self._get_num_prefill_tokens_uncached(
            request, replica_id
        )

    def on_prefill_end(self, request):
        self._pending_prefills_map[
            request.replica_id
        ] -= self._get_num_prefill_tokens_uncached(request, request.replica_id)

    """
    We find the replica which will result in the least load imbalance across all replicas,
    if the current request is assigned to it.
    """

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            load_imbalance_map = {}
            for replica_id in self._replica_schedulers.keys():
                pending_prefills = [
                    self._pending_prefills_map[replica_idx]
                    + (
                        self._get_num_prefill_tokens_uncached(request, replica_idx)
                        if replica_idx == replica_id
                        else 0
                    )
                    for replica_idx in self._replica_schedulers.keys()
                ]
                load_imbalance_map[replica_id] = max(pending_prefills) - min(
                    pending_prefills
                )
            min_load_imbalance = min(load_imbalance_map.values())
            replica_id = self._random_number_generator.choice(
                [
                    replica_id
                    for replica_id in self._replica_schedulers.keys()
                    if load_imbalance_map[replica_id] == min_load_imbalance
                ]
            )

            self._increment_pending_prefills(replica_id, request)
            request_mapping.append((replica_id, request))

        return request_mapping
