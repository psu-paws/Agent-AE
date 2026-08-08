from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.scheduler.request_queue.base_request_queue import BaseRequestQueue
from vidur.scheduler.request_queue.request_queue_registry import RequestQueueRegistry


class SarathiReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._num_running_batches = 0
        self._preempted_requests = []
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_batch_size = self._config.batch_size_cap
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def _can_allocate_request(self, request: Request) -> bool:
        is_new_request = request.id not in self._allocation_map
        if is_new_request:
            if len(self._allocation_map) >= self._max_batch_size:
                return False
            num_required_blocks = ceil(
                request.num_prefill_tokens_uncached / self._replica_config.block_size
            )
            assert num_required_blocks > 0
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )
        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            assert (
                len(self._allocation_map) < self._max_batch_size
            ), f"Cannot allocate more than {self._max_batch_size} (max_batch_size) requests"
            num_required_blocks = ceil(
                request.num_prefill_tokens_uncached / self._replica_config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = (
            self._allocation_map[request.id] * self._replica_config.block_size
        ) + request.num_prefill_tokens_cached
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,
            self._config.chunk_size - num_batch_tokens,
        )

        next_num_tokens = max(0, next_num_tokens)

        return next_num_tokens

    def _get_next_batch(self, current_time: float) -> ReplicaSchedulerOutput:
        requests: List[Request] = []
        num_tokens: List[int] = []
        skipped_requests: List[Request] = []
        running_prefills_queue: BaseRequestQueue = RequestQueueRegistry.get_from_str(
            self._request_queue._config.get_type(),
            request_queue_config=self._request_queue._config,
            execution_time_predictor=self._execution_time_predictor,
        )
        requeued_requests: List[Request] = []
        contains_prefill = False
        num_batch_tokens = 0

        # Sort the preempted requests based on the order in request queue
        # _request_queue is used here purely as a logic provider
        self._preempted_requests = self._request_queue.sort(self._preempted_requests)

        while self._preempted_requests and len(requests) < self._max_micro_batch_size:
            request = self._preempted_requests.popleft()

            if not request.is_prefill_complete:
                running_prefills_queue.push(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )
            assert next_num_tokens == 1

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop()
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue.push(victim_request)
                    requeued_requests.append(victim_request)
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue.push(request)
                    requeued_requests.append(request)
                    break
            else:
                self._allocate_request(request)
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        # TODO(nitinke): Fix assert for pipeline parallelism
        assert len(self._preempted_requests) == 0

        while len(requests) < self._max_micro_batch_size:
            is_new_prefill = False
            if len(self._request_queue) > 0 and len(running_prefills_queue) > 0:
                if running_prefills_queue.peek() < self._request_queue.peek():
                    request = running_prefills_queue.pop()
                else:
                    request = self._request_queue.pop()
                    is_new_prefill = True
            elif len(self._request_queue) > 0:
                request = self._request_queue.pop()
                is_new_prefill = True
            elif len(running_prefills_queue) > 0:
                request = running_prefills_queue.pop()
            else:
                break

            if is_new_prefill and not self._can_allocate_request(request):
                skipped_requests.append(request)
                break

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )
            if next_num_tokens == 0:
                skipped_requests.append(request)
                break

            if is_new_prefill:
                self._allocate_request(request)

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        for skipped_request in skipped_requests:
            if skipped_request.scheduled:
                self._preempted_requests.append(skipped_request)
            else:
                self._request_queue.push(skipped_request)
        skipped_requests = []
        while len(running_prefills_queue) > 0:
            request = running_prefills_queue.pop()
            assert request.num_processed_tokens > 0
            self._preempted_requests.append(request)
        return ReplicaSchedulerOutput(
            Batch(self._replica_id, requests, num_tokens) if requests else None,
            requeued_requests,
        )
