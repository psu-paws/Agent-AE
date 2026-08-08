import random
from abc import ABC, abstractmethod
from math import floor
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.logger import init_logger
from vidur.entities import Request
from vidur.entities.batch import Batch
from vidur.entities.replica import Replica
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.kv_cache.disk_kv_cache_manager import DiskKVCacheManager
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from vidur.scheduler.replica_stage_scheduler.replica_stage_scheduler import (
    ReplicaStageScheduler,
)
from vidur.types.replica_id import ReplicaId
from vidur.utils.memory_planner import MemoryPlanner
from vidur.utils.slo_manager import SLOManager

logger = init_logger(__name__)


class BaseGlobalScheduler(ABC):
    def __init__(
        self,
        config: SimulationConfig,
        replicas: Dict[ReplicaId, Replica],
    ):
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(replicas)
        self._random_number_generator = random.Random(
            config.cluster_config.global_scheduler_config.seed
        )

        cluster_config = config.cluster_config
        self._execution_time_predictors: Dict[int, object] = {}
        _group_cache_configs: Dict[int, object] = {}
        self._replica_schedulers: Dict[ReplicaId, BaseReplicaScheduler] = {}

        for i, (replica_id, replica) in enumerate(sorted(self._replicas.items())):
            group_id = cluster_config.get_replica_group(i)
            replica_config = cluster_config.get_replica_config(i)
            scheduler_config = cluster_config.get_replica_scheduler_config(i)

            # One cache config and one execution-time predictor per group, shared
            # by every replica in it.
            if group_id not in _group_cache_configs:
                cache_config = cluster_config.get_group_cache_config(group_id)
                if cache_config.num_blocks is None:
                    planner = MemoryPlanner(replica_config, cache_config)
                    cache_config.num_blocks = floor(
                        planner.get_max_kv_cache_size_in_tokens()
                        // cache_config.block_size
                    )
                _group_cache_configs[group_id] = cache_config
                self._execution_time_predictors[group_id] = (
                    ExecutionTimePredictorRegistry.get(
                        config.execution_time_predictor_config.get_type(),
                        predictor_config=config.execution_time_predictor_config,
                        replica_config=replica_config,
                        cache_config=cache_config,
                    )
                )
            cache_config = _group_cache_configs[group_id]

            self._replica_schedulers[replica_id] = ReplicaSchedulerRegistry.get_from_str(
                scheduler_config.get_type(),
                replica_config=replica_config,
                replica_scheduler_config=scheduler_config,
                request_generator_config=self._config.request_generator_config,
                cache_config=cache_config,
                request_queue_config=cluster_config.request_queue_config,
                replica=replica,
                execution_time_predictor=self._execution_time_predictors[group_id],
            )
            logger.info(
                f"Replica {i} (id={replica_id}) | "
                f"role={replica.role} | "
                f"model={replica_config.model_name} | "
                f"TP={replica_config.tensor_parallel_size}, "
                f"PP={replica_config.num_pipeline_stages} | "
                f"device={replica_config.device} | "
                f"GPU blocks={cache_config.num_blocks}, "
                f"block_size={cache_config.block_size} | "
                f"scheduler={scheduler_config.get_type()}, "
                f"chunk_size={scheduler_config.chunk_size}"
            )

        # Per-replica predictor lookup, used by events that need transfer time.
        self._replica_id_to_predictor = {
            replica_id: self._execution_time_predictors[
                cluster_config.get_replica_group(i)
            ]
            for i, replica_id in enumerate(sorted(self._replicas.keys()))
        }

        # Disk caching is from upstream functionality and is not used by this artifact.
        cc = cluster_config.cache_config
        if cc.enable_disk_caching:
            disk_kv_cache_manager = DiskKVCacheManager(
                block_size=cc.block_size,
                num_gpu_blocks=cc.disk_num_blocks,
                enable_caching=cc.enable_prefix_caching,
                caching_hash_algo=cc.prefix_caching_hash_algo,
                num_preallocate_tokens=cc.num_preallocate_tokens,
            )
            for _, replica in self._replica_schedulers.items():
                replica.set_disk_kv_cache(disk_kv_cache_manager)

        self._request_queue: List[Request] = []
        self._slo_manager = SLOManager(self._config.slo_config)

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda x: (x.arrived_at, x.id))

    def add_request(self, request: Request) -> None:
        # This is the first instance the request comes into contact with the system
        self._slo_manager.set_slos(request)
        self._request_queue.append(request)

    def on_batch_end(self, batch: Batch) -> None:
        pass

    def on_prefill_end(self, request: Request) -> None:
        pass

    def on_request_end(self, request: Request) -> None:
        pass

    def get_execution_time_predictor(self, replica_id: ReplicaId):
        return self._replica_id_to_predictor[replica_id]
    
    def get_replica_scheduler(self, replica_id: ReplicaId) -> BaseReplicaScheduler:
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(
        self, replica_id: ReplicaId, stage_id: int
    ) -> ReplicaStageScheduler:
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    @abstractmethod
    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        pass
