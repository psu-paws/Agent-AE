from vidur.entities.batch import Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.types.request_queue_type import RequestQueueType


class OrcaReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self._request_queue._config.get_type() == RequestQueueType.FCFS
        ), "Orca scheduler only supports FCFS request queues"

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._replica_config.block_size
        )
        self._max_batch_size = min(
            self._config.batch_size_cap,
            self._config.num_blocks // self._max_blocks_per_sequence,
        )
        assert (
            self._max_batch_size > 0
        ), "Not enough memory to store even a single request"

        self._num_running_batches = 0

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []

        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            if len(requests) == self._max_batch_size:
                break

            request = self._preempted_requests.popleft()
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        while self._request_queue:
            if len(requests) == self._max_batch_size:
                break

            if not self.can_allocate(self._max_blocks_per_sequence):
                break

            request = self._request_queue.popleft(0)

            self.allocate(request.id, self._max_blocks_per_sequence)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
