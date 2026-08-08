from typing import List, Optional, Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_has_slo(func):
    def wrapper(self, *args, **kwargs):
        if not self._prefill_slo_time:
            raise ValueError("Request has no slo set")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        block_hash_ids: Optional[List[int]],
        block_size: Optional[int],
        token_ids: Optional[List[int]]= None,
        session_id: Optional[int] = None,
        turn_id: Optional[int] = None,
        model_id: Optional[int] = None,
        request_id: Optional[int] = None,
        inter_request_latency: float = 0.0,
    ):
        if block_hash_ids is not None:
            last_block_size = (
                num_prefill_tokens
                + num_decode_tokens
                - (block_size * len(block_hash_ids))
            )
            assert (
                last_block_size >= 0 and last_block_size < block_size
            ), f"{last_block_size} is not in the range [0, {block_size})"

        # Trace-supplied ids are preferred so that metrics line up with the trace.
        if request_id is not None:
            self._id = int(request_id)
        else:
            self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._queued_at = arrived_at
        self._replica_id = None
        self._num_prefill_tokens = num_prefill_tokens
        self._num_prefill_tokens_cached = 0
        self._num_decode_tokens = num_decode_tokens
        self._num_processed_tokens = 0
        self._block_hash_ids = block_hash_ids
        self._token_ids = token_ids
        # token_id is optional: traces that only carry lengths (and optionally
        # block_hash_ids) are still supported, they just get coarser prefix-cache matching.
        if self._token_ids is not None:
            assert (
                len(self._token_ids)
                == self._num_prefill_tokens + self._num_decode_tokens
            )
        self._block_size = block_size
        self._session_id = session_id
        self._turn_id = turn_id
        self._model_id = model_id
        self._request_id = request_id
        self.inter_request_latency = float(inter_request_latency)

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._prefill_slo_time = None
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0
        
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0

        # Prefill-decode disaggregation tracking
        self._pd_disaggregated = False
        self._prefill_replica_id = None
        self._decode_replica_id = None
        self._decode_arrived_at = 0
        self._decode_scheduled_at = 0
        self._decode_scheduling_delay = 0

        # Decode replica pre-assigned at prefill scheduling time (PD disagg only).
        self._assigned_decode_replica_id = None
        # Count of prefill blocks already scheduled for transfer to decode.
        self._prefill_blocks_transferred: int = 0

        # Tokens actually transferred at PD handoff (prefill_tokens - decode_cached).
        self._kv_transfer_tokens: int = 0

    @property
    def replica_id(self):
        return self._replica_id

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self._num_decode_tokens if self._num_decode_tokens else 0.0

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self._num_decode_tokens if self._num_decode_tokens else 0.0

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self._num_decode_tokens if self._num_decode_tokens else 0.0

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def queued_at(self):
        return self._queued_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_prefill_tokens_cached(self) -> int:
        return self._num_prefill_tokens_cached

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        if self._num_decode_tokens == 0:
            return float("inf")
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def block_hash_ids(self) -> Optional[List[int]]:
        return self._block_hash_ids

    @property
    def token_ids(self) -> Optional[List[int]]:
        return self._token_ids

    @property
    def block_size(self) -> Optional[int]:
        return self._block_size

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    @check_has_slo
    def prefill_deadline_at(self) -> float:
        return self.queued_at + self._prefill_slo_time

    @property
    @check_has_slo
    def prefill_slo_time(self) -> float:
        return self._prefill_slo_time

    @prefill_slo_time.setter
    def prefill_slo_time(self, value: float):
        self._prefill_slo_time = value

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    @property
    def turn_id(self) -> Optional[int]:
        """Index of this request within its session (0-based)."""
        return self._turn_id

    @property
    def model_id(self) -> Optional[int]:
        return self._model_id

    @property
    def request_id(self) -> Optional[int]:
        return self._request_id

    @property
    def pd_disaggregated(self) -> bool:
        return self._pd_disaggregated

    @property
    def prefill_replica_id(self):
        return self._prefill_replica_id

    @property
    def decode_replica_id(self):
        return self._decode_replica_id

    @property
    def decode_arrived_at(self) -> float:
        return self._decode_arrived_at

    @property
    def decode_scheduled_at(self) -> float:
        return self._decode_scheduled_at

    @property
    def decode_scheduling_delay(self) -> float:
        return self._decode_scheduling_delay

    def assign_replica(
        self,
        replica_id: ReplicaId,
    ) -> None:
        assert not self._scheduled, f"Request {self._id} already scheduled."
        self._replica_id = replica_id

    def on_pd_transfer(self, time: float, decode_replica_id: ReplicaId) -> None:
        """Transfer a prefill-complete request to decode.
        Preserves _scheduled_at from the original prefill scheduling.
        Records PD-specific timing and reassigns the replica without resetting _scheduled.
        """
        assert self._is_prefill_complete
        assert self._scheduled

        self._pd_disaggregated = True
        self._prefill_replica_id = self._replica_id
        self._decode_replica_id = decode_replica_id
        self._decode_arrived_at = time

        self._replica_id = decode_replica_id

    def on_decode_batch_schedule(self, time: float) -> None:
        """Record decode scheduling time (called from on_batch_schedule)."""
        if self._pd_disaggregated and self._decode_scheduled_at == 0:
            self._decode_scheduled_at = time
            self._decode_scheduling_delay = time - self._decode_arrived_at

    def on_cache_hit(self, num_tokens_cached: int):
        assert not self._scheduled, f"Request {self._id} already scheduled."
        assert (
            self._num_processed_tokens == 0
        ), f"Request {self._id} has already processed tokens."
        assert (
            num_tokens_cached <= self._num_prefill_tokens
        ), f"Request {self._id} has {num_tokens_cached} cached tokens, but only {self._num_prefill_tokens} prefill tokens."
        self._num_processed_tokens = num_tokens_cached
        self._num_prefill_tokens_cached = num_tokens_cached

    def restart(self):
        logger.debug(f"Restarting request {self._id}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = max(self._num_prefill_tokens, self._num_processed_tokens)
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts += 1

    def on_batch_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            # For PD-transferred requests, record decode scheduling time
            self.on_decode_batch_schedule(time)
            return

        if self._num_restarts > 0:
            self._scheduled = True
            return
        # First time scheduling
        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
    ) -> None:
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time
        
        assert self._num_processed_tokens <= self.total_tokens

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            # (skip if there are no decode tokens — request completes after prefill)
            if self._num_decode_tokens > 0:
                self._num_processed_tokens += 1

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            logger.debug(f"Request {self._id} completed at {self._completed_at}")

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
    ) -> None:
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        d = {
            "id": self._id,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "arrived_at": self._arrived_at,
            "queued_at": self._queued_at,
            # "prefill_deadline_at": self.prefill_deadline_at,
            "scheduled_at": self._scheduled_at,
            "prefill_completed_at": self._prefill_completed_at,
            "completed_at": self._completed_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "num_restarts": self._num_restarts,
        }
        if self._pd_disaggregated:
            d.update({
                "pd_disaggregated": True,
                "prefill_replica_id": str(self._prefill_replica_id),
                "decode_replica_id": str(self._decode_replica_id),
                "decode_arrived_at": self._decode_arrived_at,
                "decode_scheduled_at": self._decode_scheduled_at,
                "decode_scheduling_delay": self._decode_scheduling_delay,
            })
        return d
