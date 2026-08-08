from vidur.types.base_int_enum import BaseIntEnum


class ReplicaSchedulerType(BaseIntEnum):
    FASTER_TRANSFORMER = 1
    ORCA = 2
    SARATHI = 3
    VLLM = 4
    VLLM_V1 = 5
    VLLM_V1_DISK = 6
