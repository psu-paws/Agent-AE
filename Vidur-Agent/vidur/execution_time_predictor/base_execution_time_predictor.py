from abc import ABC, abstractmethod
from typing import List

from vidur.config import BaseExecutionTimePredictorConfig, CacheConfig, ReplicaConfig
from vidur.entities import Batch, ExecutionTime
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)


class BaseExecutionTimePredictor(ABC):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        cache_config: CacheConfig,
    ) -> None:
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._cache_config = cache_config

        # get configs
        self._block_size = cache_config.block_size
        self._cache_dir = predictor_config.cache_dir
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )

    @abstractmethod
    def get_batch_execution_time(
        self, batch: Batch, pipeline_stage: int
    ) -> ExecutionTime:
        pass

    @abstractmethod
    def get_execution_time(
        self,
        execution_time_predictor_requests: List[ExecutionTimePredictorRequest],
        pipeline_stage: int,
    ) -> ExecutionTime:
        pass

    @abstractmethod
    def get_kv_transfer_time(self, num_kv_tokens: int) -> float:
        pass
