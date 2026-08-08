from abc import ABC, abstractmethod
from math import floor

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    CacheConfig,
    ReplicaConfig,
)
from vidur.config.config import BaseRequestQueueConfig
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.kv_cache.disk_kv_cache_manager import DiskKVCacheManager
from vidur.logger import init_logger
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.request_queue.base_request_queue import BaseRequestQueue
from vidur.scheduler.request_queue.request_queue_registry import RequestQueueRegistry
from vidur.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        cache_config: CacheConfig,
        request_queue_config: BaseRequestQueueConfig,
        replica: Replica,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._request_queue_config = request_queue_config
        self._cache_config = cache_config
        self._replica_id = replica.id
        self._num_stages = replica_config.num_pipeline_stages
        self._execution_time_predictor = execution_time_predictor

        # Calculate the number of blocks for the KV cache if not provided
        if not self._cache_config.num_blocks:
            memory_planner = MemoryPlanner(self._replica_config, self._cache_config)
            self._cache_config.num_blocks = floor(
                memory_planner.get_max_kv_cache_size_in_tokens()
                // self._cache_config.block_size
            )

        # Initialize the replica stage schedulers
        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == self._num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(self._num_stages)
        }
        self._num_running_batches = 0

        # Initialize the request queue
        self._waiting_queue: BaseRequestQueue = RequestQueueRegistry.get_from_str(
            self._request_queue_config.get_type(),
            request_queue_config=self._request_queue_config,
            execution_time_predictor=self._execution_time_predictor,
        )

    @property
    def replica_id(self) -> int:
        return self._replica_id

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def set_disk_kv_cache(self, disk_kv_cache_manager: DiskKVCacheManager) -> None:
        self._disk_kv_cache_manager = disk_kv_cache_manager

    @property
    @abstractmethod
    def memory_usage_percent(self) -> float:
        pass

    @abstractmethod
    def get_cached_prefill_length(self, request: Request) -> int:
        pass

    @abstractmethod
    def add_request(self, request: Request):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self, current_time: float) -> ReplicaSchedulerOutput:
        pass

    def can_schedule(self) -> bool:
        return self._num_running_batches < self._num_stages

    def on_schedule(self, current_time: float) -> ReplicaSchedulerOutput:
        assert self.can_schedule()
        replica_scheduler_output = self._get_next_batch(current_time)
        if not replica_scheduler_output.batch:
            return replica_scheduler_output
        self._num_running_batches += 1
        return replica_scheduler_output
