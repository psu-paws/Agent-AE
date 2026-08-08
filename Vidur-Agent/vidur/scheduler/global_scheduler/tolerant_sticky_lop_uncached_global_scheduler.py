from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


class TolerantStickyLOPUncachedGlobalScheduler(BaseGlobalScheduler):
    """
    Least Outstandings Prefill tokens Prefix cache aware early binding global scheduler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tolerance_factor = (
            self._config.cluster_config.global_scheduler_config.tolerance_factor
        )
        self._session_to_replica_map = {}  # sticky record tracking
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

    def _is_replica_within_imbalance_slack(
        self,
        load_imbalance_map: dict[ReplicaId, int],
        replica_id: ReplicaId,
    ):
        min_load_imbalance = min(load_imbalance_map.values())
        sticky_load_imbalance = load_imbalance_map[replica_id]
        return sticky_load_imbalance <= self._tolerance_factor * min_load_imbalance

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping = []

        while self._request_queue:
            request = self._request_queue.pop(0)
            assert request.session_id is not None, "`session_id` is required"

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
            if (
                request.session_id not in self._session_to_replica_map
                or not self._is_replica_within_imbalance_slack(
                    load_imbalance_map, self._session_to_replica_map[request.session_id]
                )
            ):
                min_load_imbalance = min(load_imbalance_map.values())
                replica_id = self._random_number_generator.choice(
                    [
                        replica_id
                        for replica_id in self._replica_schedulers.keys()
                        if load_imbalance_map[replica_id] == min_load_imbalance
                    ]
                )
                self._session_to_replica_map[request.session_id] = replica_id

            replica_id = self._session_to_replica_map[request.session_id]
            self._increment_pending_prefills(replica_id, request)
            request_mapping.append((replica_id, request))

        return request_mapping
