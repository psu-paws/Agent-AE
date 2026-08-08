from typing import Dict, List

from vidur.entities.batch import Batch, Request
from vidur.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.types.request_queue_type import RequestQueueType
from vidur.utils import cdiv


class VLLMV1ReplicaScheduler(BaseReplicaScheduler):
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

        # req_id -> Request
        self._requests: Dict[str, Request] = {}
        # self._waiting_queue has been initialized in the parent class
        self._running: List[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()

        # Decode flow control: prefill-complete requests handed off from
        # a prefill replica wait here until their whole-sequence KV fits on this
        # node, so the decode node never over-subscribes
        self._decode_pending: List[Request] = []

        # Admitted by the gate and pulling their KV from the prefill replica.
        # Holding block reservation but cannot be batched yet.
        self._transferring: List[Request] = []
        # Drained by ReplicaScheduleEvent to start each pull.
        self._newly_admitted_decode: List[Request] = []

        # session_id -> first arrival time; populated only when
        # self._config.session_priority = True.
        self._session_first_arrival: Dict[int, float] = {}

        self.sum_prefill_tokens = 0
        self.sum_kvhit_tokens = 0

        # How often the decode admission gate actually held a request back.
        self.num_decode_admissions = 0
        self.num_decode_gate_holds = 0

    def _get_queue_priority(self, request: Request):
        """Return the heap priority for a waiting request (lower = higher priority).
        Returns a tuple so that types are consistent with the starvation-aware
        reorder key. At push time we cannot detect starvation (no current_time),
        so new requests always get tier 1; the reorder pass promotes them to
        tier 0 when they have waited long enough.

        Priority modes (evaluated in order):
        - sjf_priority: Shortest remaining prefill tokens is scheduled first.
        - session_priority: first-ever arrival time of the session (session_id)
        - default (FCFS): individual request arrived_at.
        """
        need_session_arrival = (
            self._config.session_priority
            or self._config.sjf_starvation_session_fcfs
        )
        if need_session_arrival and request.session_id is not None:
            if request.session_id not in self._session_first_arrival:
                self._session_first_arrival[request.session_id] = request.arrived_at

        if self._config.sjf_priority:
            cached = self.get_cached_prefill_length(request)
            return (1, request.num_prefill_tokens - cached)

        if self._config.session_priority and request.session_id is not None:
            return (1, self._session_first_arrival[request.session_id])

        return (1, request.arrived_at)

    @property
    def memory_usage_percent(self) -> float:
        return self._kv_cache_manager.usage * 100

    def get_cached_prefill_length(self, request: Request) -> int:
        _, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(request)
        return num_computed_tokens

    def add_request(self, request: Request):
        request.assign_replica(self._replica_id)
        self._waiting_queue.push(request, self._get_queue_priority(request))
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

        # Re-sort waiting queue by remaining prefill tokens if active SJF is on.
        # Starved requests (waiting longer than sjf_starvation_timeout) are
        # promoted to the front, ordered by session or individual arrival time.
        if self._config.sjf_active_priority and len(self._waiting_queue) > 0:
            starvation_timeout = self._config.sjf_starvation_timeout
            session_fcfs = self._config.sjf_starvation_session_fcfs
            def _reorder_key(r):
                starved = starvation_timeout > 0 and (current_time - r.arrived_at) >= starvation_timeout
                if starved:
                    tiebreak = self._session_first_arrival.get(r.session_id, r.arrived_at) if session_fcfs else r.arrived_at
                    return (0, tiebreak)
                return (1, r.num_prefill_tokens - self.get_cached_prefill_length(r))
            self._waiting_queue.reorder(_reorder_key, force=starvation_timeout > 0)

        # Decode flow control (strict PD). A handed-off request is admitted only
        # once its whole sequence fits, reserving prefill + decode blocks against
        # capacity, so an admitted decode never has to preempt and recompute.
        
        # NOTE: this is an oracle policy. It reserves against num_decode_tokens,
        # which is known here only because the trace fixes the output length.
        # A real router cannot know it ahead and must instead over-admit
        # and preempt, or gated (Zero-preemption when PD hand-off).
        if self._decode_pending:
            block_size = self._cache_config.block_size
            capacity = self._kv_cache_manager.block_pool.num_gpu_blocks

            def _blocks_needed(r: Request) -> int:
                return cdiv(r.num_prefill_tokens + r.num_decode_tokens, block_size)

            # Requests still pulling their KV already hold their reservation.
            reserved = sum(
                _blocks_needed(r) for r in self._running if r.is_prefill_complete
            ) + sum(_blocks_needed(r) for r in self._transferring)
            while self._decode_pending:
                request = self._decode_pending[0]
                if reserved > 0 and reserved + _blocks_needed(request) > capacity:
                    break
                self._decode_pending.pop(0)
                self._transferring.append(request)
                self._newly_admitted_decode.append(request)
                reserved += _blocks_needed(request)
                self.num_decode_admissions += 1
            if self._decode_pending:
                self.num_decode_gate_holds += 1

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
                    self._waiting_queue.push(preempted_req, self._get_queue_priority(preempted_req))
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
                # Clip to prefill boundary: pre-computed block_hash_ids from the
                # trace can register blocks that span into decode territory.
                block_size = self._cache_config.block_size
                max_computed = (request.num_prefill_tokens // block_size) * block_size
                if num_computed_tokens > max_computed:
                    num_computed_tokens = max_computed
                    computed_blocks = computed_blocks[:num_computed_tokens // block_size]
                num_new_tokens = request.num_prefill_tokens - num_computed_tokens
                self.sum_prefill_tokens += request.num_prefill_tokens
                self.sum_kvhit_tokens += num_computed_tokens
                if num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= block_size
                    num_new_tokens = block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                assert (
                    num_new_tokens > 0
                ), f"num_new_tokens should be greater than 0 but got {num_new_tokens}"

                new_blocks = self._kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks
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
                request.on_cache_hit(num_computed_tokens)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self._config.chunk_size
        assert token_budget >= 0
        assert len(self._running) <= self._max_micro_batch_size
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_reqs) <= len(self._running)

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
        return (
            len(self._waiting_queue)
            + len(self._running)
            + len(self._decode_pending)
            + len(self._transferring)
        ) == 0

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

    def add_decode_request(self, request: Request) -> None:
        """Hold a handed-off request until its KV fits. """
        assert request.is_prefill_complete
        assert request.replica_id == self._replica_id

        self._requests[request.id] = request
        self._decode_pending.append(request)

    def pop_newly_admitted_decode(self) -> List[Request]:
        """Requests the gate admitted this step, whose KV pull now starts."""
        admitted = self._newly_admitted_decode
        self._newly_admitted_decode = []
        return admitted

    def start_decode(self, request: Request) -> None:
        """The KV has landed, so the request may now be batched for decoding."""
        self._transferring.remove(request)
        self._running.append(request)

    def remove_prefill_end_request(self, request: Request) -> None:
        """Remove request whose prefill is complete, sent to decode. """
        assert request.is_prefill_complete
        self._running.remove(request)
        self._kv_cache_manager.free(request)
        self._kv_cache_manager.free_block_hashes(request)

        assert request.id not in self.scheduled_req_ids, "on_batch_end should have removed this"
        del self._requests[request.id]