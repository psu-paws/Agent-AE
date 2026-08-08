from sklearn.ensemble import RandomForestRegressor

from vidur.config import (
    CacheConfig,
    RandomForestExecutionTimePredictorConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class RandomForestExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: RandomForestExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        cache_config: CacheConfig,
    ) -> None:
        # will trigger model training
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            cache_config=cache_config,
        )

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def _get_estimator(self):
        return RandomForestRegressor(random_state=self._config.seed)
