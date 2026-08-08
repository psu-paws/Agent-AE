from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LOP = 4
    LOP_BINARY = 5
    LOP_BATCH_END = 6
    LOP_UNCACHED = 7
    STICKY_ROUND_ROBIN = 8
    STICKY_LOR = 9
    RANKED_STICKY_LOP_UNCACHED = 10
    TOLERANT_STICKY_LOP_UNCACHED = 11
    LOAD_AWARE = 12
    DYNAMO_KV = 13
