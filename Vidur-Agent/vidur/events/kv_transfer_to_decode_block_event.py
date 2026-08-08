from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class KVTransferToDecodeBlockEvent(BaseEvent):
    """Fire at (prefill_batch_end + transfer_time) to register a prefill KV
    block on the decode replica's prefix cache, enabling future requests with
    the same prefix to skip re-transferring those blocks."""

    def __init__(
        self,
        time: float,
        request: Request,
        decode_replica_id: ReplicaId,
        block_index: int,
    ):
        super().__init__(time, EventType.KV_TRANSFER_TO_DECODE_BLOCK)
        self._request = request
        self._decode_replica_id = decode_replica_id
        self._block_index = block_index

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        decode_scheduler = global_scheduler.get_replica_scheduler(
            self._decode_replica_id
        )
        decode_scheduler._kv_cache_manager.register_brought_back_block(
            self._request, self._block_index
        )
        # print(
        #     f"[Debug] KV→decode req={self._request.id}: "
        #     f"block_idx={self._block_index} → decode {self._decode_replica_id} "
        #     f"at t={self.time:.3f}s"
        # )
        return []

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "request_id": self._request.id,
            "decode_replica_id": str(self._decode_replica_id),
            "block_index": self._block_index,
        }
