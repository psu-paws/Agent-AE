import heapq
from collections import deque
from typing import Deque, Iterable, List

from vidur.entities.request import Request
from vidur.scheduler.request_queue.base_request_queue import BaseRequestQueue
from vidur.scheduler.request_queue.prioritised_request import PrioritizedRequest


class EDFRequestQueue(BaseRequestQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_queue: List[PrioritizedRequest] = []
        self._num_prefill_tokens = 0

    def _get_prioritized_request(self, request: Request) -> PrioritizedRequest:
        return PrioritizedRequest(
            request,
            request.prefill_deadline_at,
        )

    def push(self, request):
        heapq.heappush(
            self._request_queue,
            self._get_prioritized_request(request),
        )
        self._num_prefill_tokens += request.num_prefill_tokens

    def pop(self):
        request = heapq.heappop(self._request_queue).request
        self._num_prefill_tokens -= request.num_prefill_tokens
        return request

    def peek(self):
        return self._request_queue[0]

    def to_list(self):
        return [
            prioritized_request.request for prioritized_request in self._request_queue
        ]

    def __len__(self):
        return len(self._request_queue)

    def get_num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    def sort(self, requests: Iterable[Request]) -> Deque[Request]:
        prioritised_requests = [
            self._get_prioritized_request(request) for request in requests
        ]
        prioritised_requests.sort()
        return deque(
            prioritized_request.request for prioritized_request in prioritised_requests
        )
