from collections import deque
from time import time
from typing import Deque, Dict, List, Optional

from vidur.entities.batch import Batch, Request
from vidur.kv_cache.disk_kv_cache_manager import DiskKVCacheManager
from vidur.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from vidur.logger import init_logger
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.types.request_queue_type import RequestQueueType

logger = init_logger(__name__)


class VLLMV1DiskReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self._waiting_queue._config.get_type() == RequestQueueType.FCFS
        ), "VLLM_v1 scheduler only supports FCFS request queues"
        assert (
            self._num_stages == 1
        ), "VLLM_v1 scheduler doesn't support pipeline parallelism"

        # Scheduling constraints
        self._max_batch_size = self._config.batch_size_cap
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages

        # Create the KV Cache manager
        self._kv_cache_manager = ReplicaKVCacheManager(
            block_size=self._cache_config.block_size,
            num_gpu_blocks=self._cache_config.num_blocks,
            enable_caching=self._cache_config.enable_prefix_caching,
            caching_hash_algo=self._cache_config.prefix_caching_hash_algo,
            num_preallocate_tokens=self._cache_config.num_preallocate_tokens,
        )

        self._disk_kv_cache_manager: Optional[DiskKVCacheManager] = None
        self._disk_caching = self._cache_config.enable_disk_caching

        # req_id -> Request
        self._requests: Dict[str, Request] = {}
        # self._waiting_queue has been initialized in the parent class
        self._running: List[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()

        self._log_time = time()

    @property
    def memory_usage_percent(self) -> float:
        return self._kv_cache_manager.usage * 100

    def get_cached_prefill_length(self, request: Request) -> int:
        _, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(request)
        return num_computed_tokens

    def add_request(self, request: Request):
        request.assign_replica(self._replica_id)
        self._waiting_queue.push(request)
        self._requests[request.id] = request

    def _get_request_next_num_tokens(self, request: Request, token_budget: int) -> int:
        assert not request.completed

        # Calculate `next_num_tokens`
        if request.is_prefill_complete:
            next_num_tokens = 1
        else:
            next_num_tokens = request.num_prefill_tokens - request.num_processed_tokens
        # Pass through the token budget
        next_num_tokens = min(next_num_tokens, token_budget)
        # No negative answer
        next_num_tokens = max(0, next_num_tokens)
        return next_num_tokens

    def _get_next_batch(self, current_time: float) -> ReplicaSchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_reqs: List[Request] = []
        preempted_reqs: List[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self._config.chunk_size

        # First, schedule the RUNNING requests
        req_index = 0
        while req_index < len(self._running) and token_budget > 0:
            request: Request = self._running[req_index]
            if request.id in self.scheduled_req_ids:
                req_index += 1
                continue

            # Calculate compute to do for the request
            num_new_tokens = self._get_request_next_num_tokens(request, token_budget)
            assert (
                num_new_tokens > 0
            ), "num_new_tokens should be as token_budget > 0 and request is incomplete"

            # Try to allocate memory for the request
            while True:
                new_blocks = self._kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req: Request = self._running.pop()  # from last
                    self._kv_cache_manager.free(preempted_req)
                    preempted_req.restart()
                    self._waiting_queue.push(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_reqs.append(request)
            self.scheduled_req_ids.add(request.id)
            num_scheduled_tokens[request.id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: Deque[Request] = deque()

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while len(self._waiting_queue) and token_budget > 0:
                if len(self._running) >= self._max_micro_batch_size:
                    break

                request = self._waiting_queue.peek()

                # Get already-cached tokens. `computed` means `cached` here.
                computed_blocks, num_computed_tokens = (
                    self._kv_cache_manager.get_computed_blocks(request)
                )

                disk_computed_blocks, disk_num_computed_tokens = (
                    self._disk_kv_cache_manager.get_computed_blocks(request)
                )

                num_disk_computed_blocks = len(disk_computed_blocks)
                if (
                    disk_num_computed_tokens < num_computed_tokens
                    and num_computed_tokens <= request.num_prefill_tokens
                ):
                    disk_num_computed_tokens = num_computed_tokens
                    num_disk_computed_blocks = len(computed_blocks)

                assert (
                    num_disk_computed_blocks
                    <= request.num_prefill_tokens
                    // self._disk_kv_cache_manager.block_size
                ), "Number of blocks in disk should be <= prefill tokens/block size"
                assert (
                    disk_num_computed_tokens >= num_computed_tokens
                ), f"Disk should have atleast as many tokens as replica for request {request.id}: {disk_num_computed_tokens}, {num_computed_tokens}"

                # Number of tokens to be scheduled.
                # Using `request.num_prefill_tokens` is fine even for restarted requests
                # because done decode tokens have been added to prefill tokens.
                num_new_tokens = request.num_prefill_tokens - disk_num_computed_tokens
                if num_new_tokens < 0:
                    logger.info(f"num_new_tokens is negative!")
                if num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    disk_num_computed_tokens -= self._cache_config.block_size
                    num_new_tokens = self._cache_config.block_size
                    num_disk_computed_blocks -= 1

                while (
                    request.id > 20 and len(computed_blocks) > num_disk_computed_blocks
                ):
                    logger.info(
                        f"(weird) doing this for {request.id}: {len(computed_blocks)} {num_disk_computed_blocks}"
                    )
                    num_computed_tokens -= self._cache_config.block_size
                    computed_blocks.pop()

                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                new_blocks = self._kv_cache_manager.allocate_slots_with_disk_cache(
                    request=request,
                    num_tokens=num_new_tokens,
                    num_disk_computed_blocks=num_disk_computed_blocks,
                    new_computed_blocks=computed_blocks,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                self._waiting_queue.pop()
                req_index += 1
                self._running.append(request)
                self.scheduled_req_ids.add(request.id)
                scheduled_reqs.append(request)
                assert not request.scheduled
                num_scheduled_tokens[request.id] = num_new_tokens
                token_budget -= num_new_tokens
                # Update the number of processed tokens for the request
                request.on_cache_hit(disk_num_computed_tokens)

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self._waiting_queue.extend(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self._config.chunk_size
        assert token_budget >= 0
        assert len(self._running) <= self._max_micro_batch_size
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_reqs) <= len(self._running)

        self._sync_all_requests(scheduled_reqs)

        scheduler_output = ReplicaSchedulerOutput(
            (
                Batch(
                    self._replica_id,
                    scheduled_reqs,
                    [num_scheduled_tokens[request.id] for request in scheduled_reqs],
                )
                if scheduled_reqs
                else None
            ),
            [],
        )
        # TODO(nitin): Immediately updating num_processed_tokens for the request is important for
        #  sequence pipeline parallelism and multi-step scheduling.
        # However, this is not done here to protect the invariant that num_processed_tokens is updated only after batch end.
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # for req_id, num_scheduled_token in num_scheduled_tokens.items():
        #     self._requests[req_id].num_processed_tokens += num_scheduled_token

        self.finished_req_ids = set()

        return scheduler_output

    def is_empty(self) -> bool:
        return len(self._waiting_queue) + len(self._running) == 0

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1
        new_running: List[Request] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self._running:
            req_id = request.id
            num_tokens_scheduled = batch.num_tokens_dict.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue
            elif request.completed:
                self._free_request(request)
            else:
                new_running.append(request)
            self.scheduled_req_ids.remove(req_id)
        self._running = new_running

    def _free_request(self, request: Request) -> None:
        assert request.completed
        self._kv_cache_manager.free(request)
        self._kv_cache_manager.free_block_hashes(request)
        del self._requests[request.id]

    def _sync_all_requests(self, requests: list[Request]):
        for request in requests:
            self._sync_disk_and_local_cache(request)

    def _sync_disk_and_local_cache(self, request: Request):
        if request.is_prefill_complete:
            return
        if not self._disk_caching:
            return

        blocks_in_local_cache = self._kv_cache_manager.get_allotted_blocks(request)
        blocks_in_disk_cache = self._disk_kv_cache_manager.get_allotted_blocks(request)

        num_required_blocks = min(
            blocks_in_local_cache,
            request.num_prefill_tokens // self._disk_kv_cache_manager.block_size,
        )

        if num_required_blocks <= blocks_in_disk_cache:
            return

        if num_required_blocks == 0:
            return

        if blocks_in_disk_cache == 0:
            disk_computed_blocks, _ = self._disk_kv_cache_manager.get_computed_blocks(
                request
            )
            if num_required_blocks < len(disk_computed_blocks):
                num_required_blocks = len(disk_computed_blocks)
            self._disk_kv_cache_manager.allocate_slots(
                request, num_required_blocks, disk_computed_blocks
            )
        else:
            self._disk_kv_cache_manager.allocate_slots(request, num_required_blocks)
