from vidur.types.base_int_enum import BaseIntEnum


class ExecutionTimePredictorCacheMode(BaseIntEnum):
    IGNORE_CACHE = 1  # Do not interact the cache at all
    USE_CACHE = 2  # Use cached models, predictions if they are available, else create and cache them
    REQUIRE_CACHE = 3  # Use only cached models, predictions if they are available, else raise an error
