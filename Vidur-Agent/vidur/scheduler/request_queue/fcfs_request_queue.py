import heapq
from collections import deque
from typing import Deque, List

from vidur.entities.request import Request
from vidur.scheduler.request_queue.base_request_queue import BaseRequestQueue
from vidur.scheduler.request_queue.prioritised_request import PrioritizedRequest


class FCFSRequestQueue(BaseRequestQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_queue: List[PrioritizedRequest] = []
        self._num_prefill_tokens = 0
        # reorder() is a no-op when _dirty is False, avoiding per-tick rescans.
        self._dirty: bool = False

    def _get_prioritized_request(self, request: Request) -> PrioritizedRequest:
        return PrioritizedRequest(request, (1, request.arrived_at))

    def push(self, request, priority: float = None):
        if priority is None:
            pr = self._get_prioritized_request(request)
        else:
            pr = PrioritizedRequest(request, priority)
        heapq.heappush(self._request_queue, pr)
        self._num_prefill_tokens += request.num_prefill_tokens
        self._dirty = True

    def pop(self):
        request = heapq.heappop(self._request_queue).request
        self._num_prefill_tokens -= request.num_prefill_tokens
        self._dirty = True
        return request

    def peek(self):
        return self._request_queue[0].request

    def to_list(self):
        return [
            prioritized_request.request for prioritized_request in self._request_queue
        ]

    def __len__(self):
        return len(self._request_queue)

    def get_num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    def reorder(self, priority_fn, force: bool = False) -> None:
        """ 
        Skips the rescan when no push/pop has occurred last call, 
        unless force=True (needed for time starvation timeouts)
        """
        if not self._dirty and not force:
            return
        self._request_queue = [
            PrioritizedRequest(pr.request, priority_fn(pr.request))
            for pr in self._request_queue
        ]
        heapq.heapify(self._request_queue)
        self._dirty = False

    def sort(self, requests: Deque[Request]) -> Deque[Request]:
        return deque(sorted(requests, key=lambda x: (x.arrived_at, x.id)))
