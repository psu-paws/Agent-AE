from vidur.types.base_int_enum import BaseIntEnum


class EventType(BaseIntEnum):
    # at any given time step, call the schedule event at the last
    # to ensure that all the requests are processed
    BATCH_STAGE_ARRIVAL = 1
    REQUEST_ARRIVAL = 2
    BATCH_STAGE_END = 3
    BATCH_END = 4
    PREFILL_END = 5
    REQUEST_END = 6
    GLOBAL_SCHEDULE = 7  # push requests event
    REPLICA_SCHEDULE = 8  # includes pull requests functionality
    REPLICA_STAGE_SCHEDULE = 9
    KV_BRING_BACK_BLOCK = 10
    KV_TRANSFER_TO_DECODE_BLOCK = 11
    KV_HANDOFF_COMPLETE = 12
