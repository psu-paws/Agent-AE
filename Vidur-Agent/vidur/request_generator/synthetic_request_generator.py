from collections import deque
from typing import Optional

from vidur.config import SyntheticRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from vidur.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: SyntheticRequestGeneratorConfig):
        super().__init__(config)

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self._config.length_generator_config.get_type(),
            self._config.length_generator_config,
            self._random_number_generator,
        )
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self._config.interval_generator_config.get_type(),
            self._config.interval_generator_config,
            self._random_number_generator,
        )
        self.num_sessions = self.request_length_generator.get_num_sessions()
        self.requests = [deque() for i in range (self.num_sessions)]
        self.last_arrived_at = 0
        self.num_requests_generated = [0 for i in range (self.num_sessions)]
        self.num_requests_issued = [0 for i in range (self.num_sessions)]
        self.num_turns_per_session = [len(self.request_length_generator.session_row_indices[i]) for i in range(self.num_sessions)]

        # Dep-based scheduling state (only used when trace has a dep column).
        # _remaining_deps[session_id][turn_id] = number of unsatisfied deps
        # _max_dep_completion[session_id][turn_id] = latest completion time among deps
        if self.request_length_generator.has_deps:
            self._remaining_deps = {}
            self._max_dep_completion = {}
            for session_id in range(self.num_sessions):
                self._remaining_deps[session_id] = {}
                self._max_dep_completion[session_id] = {}
                for pos, row_idx in enumerate(self.request_length_generator.session_row_indices[session_id]):
                    row = self.request_length_generator.trace_df.iloc[row_idx]
                    turn_id = int(row["turn_id"])
                    deps = row["dep"]
                    if len(deps) > 0:
                        self._remaining_deps[session_id][turn_id] = len(deps)
                        self._max_dep_completion[session_id][turn_id] = 0.0
    @property
    def is_multi_turn(self) -> bool:
        """True when at least one session contains more than one turn.
        Flat traces (one request per session) get no per-session metric stores."""
        return any(n > 1 for n in self.num_turns_per_session)

    @property
    def has_deps(self) -> bool:
        return self.request_length_generator.has_deps

    def get_dep_ready_requests(self, session_id: int, completed_turn_id: int, completion_time: float):
        """Called when a turn completes. Returns list of (Request, arrival_time) now ready."""
        ready = []
        children = self.request_length_generator.dep_children.get(session_id, {}).get(completed_turn_id, [])
        for child_turn in children:
            self._remaining_deps[session_id][child_turn] -= 1
            self._max_dep_completion[session_id][child_turn] = max(
                self._max_dep_completion[session_id][child_turn], completion_time
            )
            if self._remaining_deps[session_id][child_turn] == 0:
                length_output = self.request_length_generator.get_num_tokens_for_turn(session_id, child_turn)
                arrival_time = self._max_dep_completion[session_id][child_turn] + length_output.inter_request_latency
                request = Request(
                    arrived_at=arrival_time,
                    num_prefill_tokens=length_output.num_prefill_tokens,
                    num_decode_tokens=length_output.num_decode_tokens,
                    block_hash_ids=length_output.block_hash_ids,
                    token_ids=length_output.token_ids,
                    block_size=length_output.block_size,
                    session_id=length_output.session_id,
                    turn_id=length_output.turn_id,
                    model_id=length_output.model_id,
                    request_id=length_output.request_id,
                    inter_request_latency=length_output.inter_request_latency,
                )
                self.num_requests_generated[session_id] += 1
                self.num_requests_issued[session_id] += 1
                ready.append((request, arrival_time))
        return ready

    # Attempt to generate a new request and append it to the queue of requests
    def _generate_next_request(self, session_id, next_query_arrival=None) -> None:
        if self.num_requests_generated[session_id] >= self.num_turns_per_session[session_id]:
            return

        # With dep-based scheduling, only the turn-0 request (dep=[]) is generated here via Poisson. 
        # All subsequent turns are released through get_dep_ready_requests when their dependencies complete.
        if self.has_deps and self.num_requests_generated[session_id] > 0:
            return

        # Gating if num_requests limited
        if self._config.num_requests is not None:
            if sum(self.num_requests_generated) >= self._config.num_requests:
                return

        if self._config.duration is not None:
            if self.last_arrived_at >= self._config.duration:
                return

        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        assert isinstance(inter_request_time, float)
        request_length_output = self.request_length_generator.get_next_num_tokens(session_id)
        if self.num_requests_generated[session_id] == 0:
            # First turn of a session: advance the shared arrival clock by the sampled inter-arrival time.
            # Later turns are chained off the previous turn's completion, and the caller passes that time
            if next_query_arrival is None:
                self.last_arrived_at = self.last_arrived_at + inter_request_time
        if next_query_arrival is not None:
            self.last_arrived_at = next_query_arrival
        self.num_requests_generated[session_id] += 1
        self.requests[session_id].append(
            Request(
                arrived_at=self.last_arrived_at,
                num_prefill_tokens=request_length_output.num_prefill_tokens,
                num_decode_tokens=request_length_output.num_decode_tokens,
                block_hash_ids=request_length_output.block_hash_ids,
                token_ids=request_length_output.token_ids,
                block_size=request_length_output.block_size,
                session_id=request_length_output.session_id,
                turn_id=request_length_output.turn_id,
                model_id=request_length_output.model_id,
                request_id=request_length_output.request_id,
                inter_request_latency=request_length_output.inter_request_latency,
            )
        )

    def request_arrival_left(self) -> Optional[float]:
        return 1 if sum(self.num_requests_generated) < sum(self.num_turns_per_session) else None

    def get_next_request_arrival_time(self, session_id, consecutive_query_arrival=None) -> Optional[float]:
        if session_id >= self.num_sessions:
            return None
        if len(self.requests[session_id]) == 0:
            self._generate_next_request(session_id, consecutive_query_arrival)

        return self.requests[session_id][0].arrived_at if len(self.requests[session_id]) > 0 else None

    def get_next_request(self, session_id) -> Optional[Request]:
        if len(self.requests[session_id]) == 0:
            self._generate_next_request(session_id)
        self.num_requests_issued[session_id] += 1

        return self.requests[session_id].popleft() if len(self.requests[session_id]) > 0 else None
