from vidur.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from vidur.execution_time_predictor.random_forest_execution_time_predictor import (
    RandomForestExecutionTimePredictor,
)
from vidur.types import ExecutionTimePredictorType
from vidur.utils.base_registry import BaseRegistry


class ExecutionTimePredictorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ExecutionTimePredictorType:
        return ExecutionTimePredictorType.from_str(key_str)


ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.RANDOM_FOREST, RandomForestExecutionTimePredictor
)
ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.LINEAR_REGRESSION, LinearRegressionExecutionTimePredictor
)
