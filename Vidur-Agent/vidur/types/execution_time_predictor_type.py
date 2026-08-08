from vidur.types.base_int_enum import BaseIntEnum


class ExecutionTimePredictorType(BaseIntEnum):
    DUMMY = 1
    RANDOM_FOREST = 2
    LINEAR_REGRESSION = 3
