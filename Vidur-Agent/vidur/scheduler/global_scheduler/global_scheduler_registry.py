from vidur.scheduler.global_scheduler.lop_batch_end_global_scheduler import (
    LOPBatchEndGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lop_binary_global_scheduler import (
    LOPBinaryGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lop_global_scheduler import LOPGlobalScheduler
from vidur.scheduler.global_scheduler.dynamo_kv_global_scheduler import (
    DynamoStyleKVGlobalScheduler,
)
from vidur.scheduler.global_scheduler.load_aware_global_scheduler import (
    LoadAwareGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lop_uncached_global_scheduler import (
    LOPUncachedGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.ranked_sticky_lop_uncached_global_scheduler import (
    RankedStickyLOPUncachedGlobalScheduler,
)
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from vidur.scheduler.global_scheduler.sticky_lor_scheduler import (
    StickyLORGlobalScheduler,
)
from vidur.scheduler.global_scheduler.sticky_round_robin_global_scheduler import (
    StickyRoundRobinGlobalScheduler,
)
from vidur.scheduler.global_scheduler.tolerant_sticky_lop_uncached_global_scheduler import (
    TolerantStickyLOPUncachedGlobalScheduler,
)
from vidur.types import GlobalSchedulerType
from vidur.utils.base_registry import BaseRegistry


class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOP, LOPGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.LOP_BINARY, LOPBinaryGlobalScheduler
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.LOP_BATCH_END, LOPBatchEndGlobalScheduler
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.LOP_UNCACHED, LOPUncachedGlobalScheduler
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.STICKY_ROUND_ROBIN, StickyRoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.STICKY_LOR, StickyLORGlobalScheduler
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.RANKED_STICKY_LOP_UNCACHED,
    RankedStickyLOPUncachedGlobalScheduler,
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.TOLERANT_STICKY_LOP_UNCACHED,
    TolerantStickyLOPUncachedGlobalScheduler,
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.LOAD_AWARE,
    LoadAwareGlobalScheduler,
)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.DYNAMO_KV,
    DynamoStyleKVGlobalScheduler,
)
