from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: ReplicaId, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.prefill_end_event import PrefillEndEvent
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        from vidur.events.request_end_event import RequestEndEvent

        self._batch.on_batch_end(self.time)
        global_scheduler.on_batch_end(self._batch)
        replica_scheduler = global_scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        # Attempt to pull requests for the next batch
        next_events = [ReplicaScheduleEvent(self.time, self._replica_id)]
        for request in self._batch.completed_prefills:
            next_events.append(PrefillEndEvent(self.time, request))
        for request in self._batch.completed_requests:
            next_events.append(RequestEndEvent(self.time, request))

        # For PD prefill-phase requests: stream newly completed KV blocks to the
        # pre-assigned decode replica as each prefill chunk finishes. 
        # (assumption) Transfer is overlapped with prefill, only the last chunk handoff is exposed.
        if getattr(global_scheduler, '_prefill_replica_to_model', None):
            from vidur.events.kv_transfer_to_decode_block_event import KVTransferToDecodeBlockEvent

            block_size = (
                global_scheduler._config.cluster_config.cache_config.block_size
            )
            for request in self._batch.requests:
                decode_rid = getattr(request, '_assigned_decode_replica_id', None)
                if (
                    decode_rid is not None
                    and not request.is_prefill_complete
                    and not request.completed
                ):
                    current_blocks = request.num_processed_tokens // block_size
                    last_sent = request._prefill_blocks_transferred
                    if current_blocks > last_sent:
                        predictor = global_scheduler.get_execution_time_predictor(
                            request.replica_id
                        )
                        cross_node_pairs = getattr(global_scheduler, '_cross_node_pairs', {})
                        decode_tp = cross_node_pairs.get(
                            (request.replica_id, decode_rid), 0
                        )
                        for block_idx in range(last_sent, current_blocks):
                            if decode_tp > 0:
                                T_ms = predictor.get_kv_transfer_time_cross_node(block_size, decode_tp)
                            else:
                                T_ms = predictor.get_kv_transfer_time(block_size)
                            next_events.append(KVTransferToDecodeBlockEvent(
                                self.time + T_ms / 1e3,
                                request,
                                decode_rid,
                                block_idx,
                            ))
                        request._prefill_blocks_transferred = current_blocks

        # KV generated at Decode brought back to Prefill, inspired by 
        # https://docs.vllm.ai/en/stable/features/nixl_connector_usage/#bidirectional-kv-transfer-multi-turn
        if global_scheduler._config.kv_cache_bring_back:
            from vidur.events.kv_bring_back_block_event import KVBringBackBlockEvent

            block_size = (
                global_scheduler._config.cluster_config.cache_config.block_size
            )
            for request in self._batch.requests:
                if (
                    request.is_prefill_complete
                    and request.pd_disaggregated
                    and not request.completed
                    and request.num_processed_tokens % block_size == 0
                ):
                    block_index = request.num_processed_tokens // block_size - 1
                    predictor = global_scheduler.get_execution_time_predictor(
                        request.prefill_replica_id
                    )
                    cross_node_pairs = getattr(global_scheduler, '_cross_node_pairs', {})
                    decode_tp = cross_node_pairs.get(
                        (request.prefill_replica_id, request.decode_replica_id), 0
                    )
                    if decode_tp > 0:
                        T_block_ms = predictor.get_kv_transfer_time_cross_node(block_size, decode_tp)
                    else:
                        T_block_ms = predictor.get_kv_transfer_time(block_size)
                    next_events.append(
                        KVBringBackBlockEvent(
                            self.time + T_block_ms / 1e3,
                            request,
                            request.prefill_replica_id,
                            block_index,
                        )
                    )

        return next_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "batch_id": self._batch.id,
        }
