import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Dict, List, Optional

from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.device_sku_config import BaseDeviceSKUConfig
from vidur.config.flat_dataclass import create_flat_dataclass
from vidur.config.model_config import BaseModelConfig
from vidur.config.node_sku_config import BaseNodeSKUConfig
from vidur.config.utils import dataclass_to_dict
from vidur.logger import init_logger
from vidur.types import (
    ExecutionTimePredictorCacheMode,
    ExecutionTimePredictorType,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
    RequestQueueType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    @staticmethod
    @abstractmethod
    def get_type():
        pass


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum tokens."},
    )

    @staticmethod
    @abstractmethod
    def get_type():
        pass


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class UniformRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second"},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.UNIFORM


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )
    preserve_request_order: bool = field(
        default=True,
        metadata={
            "help": "Preserve request order for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )

    @staticmethod
    @abstractmethod
    def get_type():
        pass


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )
    num_requests: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE


@dataclass
class SloConfig:
    prefill_e2e_time_normalized: Optional[float] = field(
        default=None,
        metadata={"help": "Target Normalized TTFT"},
    )
    prefill_e2e_time_min: Optional[float] = field(
        default=None,
        metadata={"help": "Target Min TTFT"},
    )


@dataclass
class CacheConfig:
    block_size: int = field(
        default=16,
        metadata={"help": "Block size."},
    )
    num_blocks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks."},
    )
    watermark_blocks_fraction: float = field(
        default=0.01,
        metadata={"help": "Watermark blocks fraction."},
    )
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},
    )
    enable_prefix_caching: bool = field(
        default=False,
        metadata={"help": "Enable prefix caching."},
    )
    prefix_caching_hash_algo: str = field(
        default="builtin",
        metadata={"help": "Prefix caching hash algorithm."},
    )
    num_preallocate_tokens: int = field(
        default=64,
        metadata={"help": "Number of preallocate tokens."},
    )
    enable_disk_caching: bool = field(
        default=False, metadata={"help": "Enable caching to disk."}
    )
    disk_num_blocks: int = field(
        default=sys.maxsize, metadata={"help": "Number of blocks in disk cache."}
    )


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):
    batch_size_cap: int = field(
        default=128,
        metadata={"help": "Maximum batch size cap."},
    )

    @staticmethod
    @abstractmethod
    def get_type():
        pass


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for vLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM


@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for Sarathi."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI


@dataclass
class VllmV1SchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=4096,
        metadata={"help": "Chunk size for chunked prefill."},
    )
    session_priority: bool = field(
        default=False,
        metadata={"help": "When True, timed-out (starved) requests are ordered by session_id instead of arrived_at, so older sessions are served first rather than the individual request that arrived earliest."},
    )
    sjf_priority: bool = field(
        default=False,
        metadata={"help": "When True, waiting requests are ordered at push time by remaining prefill tokens (num_prefill_tokens - prefix_cache_hit_tokens), implementing Shortest Job First. Takes precedence over session_priority."},
    )
    sjf_active_priority: bool = field(
        default=False,
        metadata={"help": "When True, the waiting queue is re-sorted by remaining prefill tokens at the start of every scheduling round, capturing cache state changes between rounds. Can be combined with sjf_priority."},
    )
    sjf_starvation_timeout: float = field(
        default=0.0,
        metadata={"help": "Starvation prevention for SJF: if a waiting request's age (current_time - arrived_at) exceeds this value (in seconds), it is promoted to the front of the queue. 0 disables starvation prevention."},
    )
    sjf_starvation_session_fcfs: bool = field(
        default=False,
        metadata={"help": "Controls tiebreak ordering among starved requests. False (default): FCFS by individual request arrived_at. True: FCFS by session first arrival time, grouping turns of the same session together."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM_V1


@dataclass
class VllmV1DiskSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for chunked prefill."},
    )
    # TODO: V1DiskScheduler flag added, but no changes made in disk scheduler.
    session_priority: bool = field(
        default=False,
        metadata={"help": "When True, requests sharing the same request_id (multi-turn session) are prioritized by the session's first-ever arrival time rather than each request's individual arrival time."},
    )
    sjf_priority: bool = field(
        default=False,
        metadata={"help": "When True, waiting requests are ordered at push time by remaining prefill tokens (num_prefill_tokens - prefix_cache_hit_tokens), implementing Shortest Job First. Takes precedence over session_priority."},
    )
    sjf_active_priority: bool = field(
        default=False,
        metadata={"help": "When True, the waiting queue is re-sorted by remaining prefill tokens at the start of every scheduling round, capturing cache state changes between rounds. Can be combined with sjf_priority."},
    )
    sjf_starvation_timeout: float = field(
        default=0.0,
        metadata={"help": "Starvation prevention for SJF: if a waiting request's age (current_time - arrived_at) exceeds this value (in seconds), it is promoted to the front of the queue. 0 disables starvation prevention."},
    )
    sjf_starvation_session_fcfs: bool = field(
        default=False,
        metadata={"help": "Controls tiebreak ordering among starved requests. False (default): FCFS by individual request arrived_at. True: FCFS by session first arrival time, grouping turns of the same session together."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM_V1_DISK


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},
    )
    enable_chrome_trace: bool = field(
        default=False,
        metadata={"help": "Enable Chrome tracing."},
    )
    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},
    )
    store_plots: bool = field(
        default=False,
        metadata={"help": "Whether to store plots."},
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep individual batch metrics. This can lead to large disk usage."
        },
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},
    )
    output_dir: str = field(
        default="simulator_output",
        metadata={"help": "Output directory."},
    )
    no_timestamp: bool = field(
        default=False,
        metadata={"help": "If set, use output_dir as-is without appending a timestamp."},
    )

    def __post_init__(self):
        if not self.no_timestamp:
            self.output_dir = (
                f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            )
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ReplicaConfig:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Model name."},
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    pd_disaggregation: bool = field(
        default=False,
        metadata={
            "help": "Whether this replica participates in prefill-decode disaggregation."
        },
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},
    )

    def __post_init__(self):
        self.world_size = self.num_pipeline_stages * self.tensor_parallel_size
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )

        assert self.model_config.num_q_heads % self.tensor_parallel_size == 0
        assert self.model_config.num_layers % self.num_pipeline_stages == 0
        assert self.model_config.embedding_dim % self.tensor_parallel_size == 0
        assert self.model_config.embedding_dim % self.model_config.num_q_heads == 0

        self._num_layers_per_pipeline_stage = (
            self.model_config.num_layers // self.num_pipeline_stages
        )
        self._attention_head_dim = (
            self.model_config.embedding_dim // self.model_config.num_q_heads
        )
        self._q_heads_per_tensor_parallel_worker = (
            self.model_config.num_q_heads // self.tensor_parallel_size
        )
        self._kv_heads_per_tensor_parallel_worker = ceil(
            self.model_config.num_kv_heads / self.tensor_parallel_size
        )

    @property
    def num_layers_per_pipeline_stage(self):
        return self._num_layers_per_pipeline_stage

    @property
    def attention_head_dim(self):
        return self._attention_head_dim

    @property
    def q_heads_per_tensor_parallel_worker(self):
        return self._q_heads_per_tensor_parallel_worker

    @property
    def kv_heads_per_tensor_parallel_worker(self):
        return self._kv_heads_per_tensor_parallel_worker


@dataclass
class BaseRequestQueueConfig(BasePolyConfig):
    @staticmethod
    @abstractmethod
    def get_type() -> RequestQueueType:
        pass


@dataclass
class FCFSRequestQueueConfig(BaseRequestQueueConfig):
    @staticmethod
    def get_type():
        return RequestQueueType.FCFS


@dataclass
class EDFRequestQueueConfig(BaseRequestQueueConfig):
    @staticmethod
    def get_type():
        return RequestQueueType.EDF


@dataclass
class BaseGlobalSchedulerConfig(BasePolyConfig):
    seed: int = field(
        default=67,
        metadata={
            "help": "Seed for the random number generator to be used only in routers."
        },
    )

    @staticmethod
    @abstractmethod
    def get_type():
        pass


@dataclass
class RandomGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM


@dataclass
class RoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.ROUND_ROBIN


@dataclass
class LoadAwareGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    use_load_aware_routing: bool = field(
        default=True,
        metadata={
            "help": (
                "When True, prefill replica selection uses load-aware scoring "
                "(running load + shadow tokens + KV-cache-adjusted effective tokens). "
                "When False, falls back to round-robin. Automatically skips scoring "
                "when a model's prefill pool has only one replica."
            )
        },
    )
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOAD_AWARE


@dataclass
class DynamoStyleKVGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    """Dynamo-style KV-router scoring"""

    overlap_score_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight on the prefill term relative to decode load in the "
            "routing cost, matching Dynamo's knob of the same name. Higher values "
            "favour prefix-cache reuse; 0.0 gives pure decode load balancing."
        },
    )

    @staticmethod
    def get_type():
        return GlobalSchedulerType.DYNAMO_KV



@dataclass
class LORGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOR


@dataclass
class LOPGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOP


@dataclass
class LOPBinaryGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOP_BINARY


@dataclass
class LOPBatchEndGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOP_BATCH_END


@dataclass
class LOPUncachedGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOP_UNCACHED


@dataclass
class StickyRoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.STICKY_ROUND_ROBIN


@dataclass
class StickyLorGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.STICKY_LOR


@dataclass
class TolerantStickyLopUncachedGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    tolerance_factor: float = field(
        default=2.0,
        metadata={"help": "Imbalance tolerance factor."},
    )

    @staticmethod
    def get_type():
        return GlobalSchedulerType.TOLERANT_STICKY_LOP_UNCACHED


@dataclass
class RankedStickyLopUncachedGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    top_k: int = field(
        default=2,
        metadata={"help": "Top K for the ranked sticky LOP uncached global scheduler."},
    )

    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANKED_STICKY_LOP_UNCACHED


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={
            "help": "Seed for making operation runtime predictions reproducible (best effort)."
        },
    )
    compute_input_file: str = field(
        default="../../data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv",
        metadata={"help": "Path to the compute input file."},
    )
    attention_input_file: str = field(
        default="../../data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},
    )
    all_reduce_input_file: str = field(
        default="../../data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},
    )
    send_recv_input_file: str = field(
        default="../../data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},
    )
    cpu_overhead_input_file: str = field(
        default="../../data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},
    )
    cache_mode: str = field(
        default=ExecutionTimePredictorCacheMode.USE_CACHE,
        metadata={"help": "Cache access mode for the execution time predictor."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},
    )
    prefill_chunk_size_prediction_granularity: int = field(
        default=32,
        metadata={"help": "Prefill chunk size prediction granularity."},
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},
    )
    prediction_max_batch_size: int = field(
        default=512,
        metadata={"help": "Max batch size for prediction."},
    )
    prediction_max_tokens_per_request: int = field(
        default=600000,
        metadata={"help": "Max tokens per request for prediction."},
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.0,
        metadata={"help": "Attention decode batching overhead fraction."},
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.0,
        metadata={"help": "Attention prefill batching overhead fraction."},
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.00,
        metadata={"help": "NCCL CPU launch overhead in ms."},
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},
    )

    def __post_init__(self):
        self._location = os.path.dirname(os.path.abspath(__file__))
        self.compute_input_file = os.path.join(self._location, self.compute_input_file)
        self.attention_input_file = os.path.join(
            self._location, self.attention_input_file
        )
        self.all_reduce_input_file = os.path.join(
            self._location, self.all_reduce_input_file
        )
        self.send_recv_input_file = os.path.join(
            self._location, self.send_recv_input_file
        )
        self.cpu_overhead_input_file = os.path.join(
            self._location, self.cpu_overhead_input_file
        )
        self.cache_mode = ExecutionTimePredictorCacheMode.from_str(self.cache_mode)


@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class RandomForestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FOREST


_REPLICA_SCHEDULER_CONFIG_MAP = {
    "vllm_v1": (VllmV1SchedulerConfig, ReplicaSchedulerType.VLLM_V1),
    "vllm_v1_disk": (VllmV1DiskSchedulerConfig, ReplicaSchedulerType.VLLM_V1_DISK),
    "sarathi": (SarathiSchedulerConfig, ReplicaSchedulerType.SARATHI),
    "vllm": (VllmSchedulerConfig, ReplicaSchedulerType.VLLM),
    "orca": (OrcaSchedulerConfig, ReplicaSchedulerType.ORCA),
    "faster_transformer": (FasterTransformerSchedulerConfig, ReplicaSchedulerType.FASTER_TRANSFORMER),
}


@dataclass
class ClusterConfig:
    num_replicas: int = field(
        default=1,
        metadata={"help": "Number of replicas."},
    )
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    cache_config: CacheConfig = field(
        default_factory=CacheConfig,
        metadata={"help": "Cache config."},
    )
    global_scheduler_config: BaseGlobalSchedulerConfig = field(
        default_factory=RoundRobinGlobalSchedulerConfig,
        metadata={"help": "Global scheduler config."},
    )
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig,
        metadata={"help": "Replica scheduler config."},
    )
    request_queue_config: BaseRequestQueueConfig = field(
        default_factory=FCFSRequestQueueConfig,
        metadata={"help": "Request queue config."},
    )
    replica_groups_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON file defining heterogeneous replica groups. "
                  "Overrides num_replicas and replica_config when set."},
    )

    def __post_init__(self):
        # internal heterogeneous mappings (populated from JSON)
        self._replica_id_to_config: Dict[int, ReplicaConfig] = {}
        self._replica_id_to_role: Dict[int, str] = {}
        self._replica_id_to_group: Dict[int, int] = {}
        self._replica_id_to_scheduler_config: Dict[int, BaseReplicaSchedulerConfig] = {}
        self._group_configs: Dict[int, ReplicaConfig] = {}
        # Per-group cache config overrides (only fields specified in JSON)
        self._group_cache_overrides: Dict[int, dict] = {}
        self._pd_pools: List[Dict] = []
        
        if self.replica_groups_config is not None:
            self._parse_replica_groups()

    def _parse_replica_groups(self):
        with open(self.replica_groups_config) as f:
            data = json.load(f)

        groups = data["replica_groups"]
        replica_index = 0
        for group_id, group in enumerate(groups):
            role = group.get("role", "agg")
            count = group.get("num_replicas", 1)
            rc_dict = group.get("replica_config", {})
            sched_dict = group.get("replica_scheduler_config", {})

            # Per-group cache config overrides (only disk-related fields for now)
            cc_dict = group.get("cache_config", {})
            self._group_cache_overrides[group_id] = cc_dict

            # Build ReplicaConfig for this group
            rc = ReplicaConfig(**rc_dict)
            self._group_configs[group_id] = rc

            # Build ReplicaSchedulerConfig for this group.
            sched_type = sched_dict.pop("type", None)
            if sched_type and sched_type in _REPLICA_SCHEDULER_CONFIG_MAP:
                config_cls, _ = _REPLICA_SCHEDULER_CONFIG_MAP[sched_type]
                cli_cfg = self.replica_scheduler_config
                if isinstance(cli_cfg, config_cls):
                    # Merge: start from CLI config, override with JSON fields
                    import dataclasses
                    base = dataclasses.asdict(cli_cfg)
                    base.update(sched_dict)
                    sched_config = config_cls(**base)
                else:
                    sched_config = config_cls(**sched_dict)
            else:
                # Fall back to the CLI-provided scheduler config
                sched_config = self.replica_scheduler_config

            for _ in range(count):
                self._replica_id_to_config[replica_index] = rc
                self._replica_id_to_role[replica_index] = role
                self._replica_id_to_group[replica_index] = group_id
                self._replica_id_to_scheduler_config[replica_index] = sched_config
                replica_index += 1

        # Override num_replicas with total
        self.num_replicas = replica_index

        # Override the top-level replica_config so serialized config.json
        # reflects the actual model used (first group's config).
        if self._group_configs:
            self.replica_config = self._group_configs[0]

        # Parse explicit pool format:
        #   "replica_groups_pools": [
        #     {"prefill": [0, 1, 2], "decode": [4, 5]},   <- model_id 0 (array index)
        #     {"prefill": [3],       "decode": [6, 7]}    <- model_id 1 (array index)
        #   ]
        # model_id is inferred from the array index so it always aligns with
        # request._model_id (0-indexed) from the trace — no manual field needed.
        # prefill/decode values are replica indices.
        # Asymmetric numbers of prefill vs decode nodes are allowed.
        for mid, pool_def in enumerate(data.get("replica_groups_pools", [])):
            self._pd_pools.append({
                "model_id": mid,
                "prefill":  [int(i) for i in pool_def["prefill"]],
                "decode":   [int(i) for i in pool_def["decode"]],
                # Whether prefill and decode sit on different nodes.
                # analytic inter-node bandwidth model for KV handoff.
                "cross_node": bool(pool_def.get("cross_node", False)),
            })

        logger.info(
            f"Loaded {len(groups)} replica groups, "
            f"{self.num_replicas} total replicas, "
            f"{len(self._pd_pools)} explicit pools"
        )

    @property
    def is_heterogeneous(self) -> bool:
        return len(self._replica_id_to_config) > 0

    def get_replica_config(self, replica_index: int) -> "ReplicaConfig":
        if not self.is_heterogeneous:
            return self.replica_config
        return self._replica_id_to_config[replica_index]

    def get_replica_role(self, replica_index: int) -> str:
        if not self.is_heterogeneous:
            return "agg"
        return self._replica_id_to_role[replica_index]

    def get_replica_group(self, replica_index: int) -> int:
        if not self.is_heterogeneous:
            return 0
        return self._replica_id_to_group[replica_index]

    def get_replica_scheduler_config(self, replica_index: int) -> "BaseReplicaSchedulerConfig":
        if not self.is_heterogeneous:
            return self.replica_scheduler_config
        return self._replica_id_to_scheduler_config[replica_index]

    def get_group_cache_config(self, group_id: int) -> "CacheConfig":
        import copy as _copy
        cc = _copy.copy(self.cache_config)
        overrides = self._group_cache_overrides.get(group_id, {})
        for field_name, value in overrides.items():
            if hasattr(cc, field_name):
                setattr(cc, field_name, value)
        return cc


@dataclass
class SimulationConfig(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    time_limit: int = field(
        default=0,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},
    )
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster config."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "Request generator config."},
    )
    kv_cache_bring_back: bool = field(
        default=False,
        metadata={
            "help": (
                "KV bring-back: asynchronously transfer each completed decode KV block "
                "back to the prefill replica so it can be reused as a prefix cache hit "
                "for future requests with the same decode prefix. inspired by "
                "https://docs.vllm.ai/en/stable/features/nixl_connector_usage/#bidirectional-kv-transfer-multi-turn"
            )
        },
    )

    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics config."},
    )
    slo_config: SloConfig = field(
        default_factory=SloConfig,
        metadata={"help": "SLO config."},
    )

    def __post_init__(self):
        self.write_config_to_file()

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
