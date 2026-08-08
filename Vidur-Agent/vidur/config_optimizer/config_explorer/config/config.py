import hashlib
from dataclasses import dataclass
from itertools import product
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str
    identifier: str
    exclude_tp_dims: List[int] = None

    def get_key(self):
        return self.name

    def to_config_dict(self):
        return {
            "replica_config_model_name": self.identifier,
        }

    def is_tensor_parallel_degree_valid(self, tp_degree: int):
        return self.exclude_tp_dims is None or tp_degree not in self.exclude_tp_dims


@dataclass
class TraceConfig:
    name: str
    trace_file: str
    max_seq_len: int
    enable_prefix_caching: bool

    def get_key(self):
        return f"{self.name}_tk{self.max_seq_len}"

    def to_config_dict(self):
        return {
            "request_generator_config_type": "synthetic",
            "length_generator_config_type": "trace",
            "trace_request_length_generator_config_trace_file": self.trace_file,
            "trace_request_length_generator_config_max_tokens": self.max_seq_len,
            "trace_request_length_generator_config_prefill_scale_factor": 1,
            "trace_request_length_generator_config_decode_scale_factor": 1,
            "vllm_scheduler_config_max_tokens_in_batch": self.max_seq_len,
            "cache_config_enable_prefix_caching": None,
        }


@dataclass
class ClusterConfig:
    device: str
    network_device: str
    gpus_per_node: int

    def get_key(self):
        return self.network_device

    def to_config_dict(self):
        return {
            "replica_config_device": self.device,
            "replica_config_network_device": self.network_device,
        }


@dataclass
class GlobalSchedulerConfig:
    scheduler: str
    top_k: Optional[int] = None
    tolerance_factor: Optional[float] = None

    def get_key(self):
        key = self.scheduler
        if self.top_k is not None:
            key += f"_topk{self.top_k}"
        if self.tolerance_factor is not None:
            key += f"_{self.tolerance_factor}x"
        return key

    def to_config_dict(self):
        if self.scheduler == "ranked_sticky_lop_uncached":
            assert self.top_k is not None
            return {
                "global_scheduler_config_type": self.scheduler,
                "ranked_sticky_lop_uncached_global_scheduler_config_top_k": self.top_k,
            }
        elif self.scheduler == "tolerant_sticky_lop_uncached":
            assert self.tolerance_factor is not None
            return {
                "global_scheduler_config_type": self.scheduler,
                "tolerant_sticky_lop_uncached_global_scheduler_config_tolerance_factor": self.tolerance_factor,
            }
        else:
            return {
                "global_scheduler_config_type": self.scheduler,
            }


@dataclass
class RequestQueueConfig:
    name: str
    provider: str
    slo_prefill_e2e_time_normalized: Optional[float] = None

    def get_key(self):
        return self.name

    def to_config_dict(self):
        config_dict = {
            "request_queue_config_type": self.provider,
        }
        if self.provider == "edf":
            config_dict["slo_config_prefill_e2e_time_normalized"] = (
                self.slo_prefill_e2e_time_normalized
            )
        return config_dict


@dataclass
class ReplicaSchedulerConfig:
    scheduler: str
    chunk_size: Optional[int] = None

    def get_key(self):
        scheduler = self.scheduler

        if self.chunk_size is not None:
            scheduler += f"_cs{self.chunk_size}"

        return scheduler

    def to_config_dict(self):
        if self.scheduler == "vllm":
            return {
                "replica_scheduler_config_type": "vllm",
            }
        elif self.scheduler == "sarathi":
            assert self.chunk_size is not None
            return {
                "replica_scheduler_config_type": self.scheduler,
                "sarathi_scheduler_config_chunk_size": self.chunk_size,
            }
        elif self.scheduler == "vllm_v1":
            assert self.chunk_size is not None
            return {
                "replica_scheduler_config_type": "vllm_v1",
                "vllm_v1_scheduler_config_chunk_size": self.chunk_size,
            }
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")


@dataclass
class SloConfig:
    metric: str
    value: float


@dataclass
class SloConfigs:
    name: str
    type: str
    quantile: float
    slos: List[SloConfig]

    def __post_init__(self):
        slo_types = ["and", "or"]
        if self.type not in slo_types:
            raise ValueError(
                f"Invalid slo type: {self.type}, allowed values are {slo_types}"
            )
        self.slos = [SloConfig(**slo) for slo in self.slos]

    def get_key(self):
        return self.name


@dataclass
class SearchConfig:
    trace: str
    model: str
    search_for: str
    num_requests: int
    num_replicas: int
    qps: float
    qps_multipliers: Optional[List[float]] = None

    def __post_init__(self):
        if self.search_for not in ["qps", "num_replicas", "isoqps"]:
            raise ValueError(f"Invalid search_for value: {self.search_for}")
        if self.search_for == "isoqps":
            assert len(self.qps_multipliers) > 0
        else:
            assert self.qps_multipliers is None or len(self.qps_multipliers) == 0

    def get_key(self):
        if self.search_for == "num_replicas":
            return f"{self.search_for}_r{self.num_replicas}_reqs{self.num_requests}"
        else:
            # For isoqps search, we group runs with different multipliers together
            return f"{self.search_for}_q{self.qps}_reqs{self.num_requests}"


class JobConfig:
    def __init__(
        self,
        search_config: SearchConfig,
        model_config: ModelConfig,
        cluster_config: ClusterConfig,
        trace_config: TraceConfig,
        global_scheduler_config: GlobalSchedulerConfig,
        request_queue_config: RequestQueueConfig,
        replica_scheduler_config: ReplicaSchedulerConfig,
        slo_configs: SloConfigs,
        num_tensor_parallel_workers: int,
        num_pipeline_stages: int,
        batch_size: int,
    ):
        self.model_config = model_config
        self.cluster_config = cluster_config
        self.trace_config = trace_config
        self.global_scheduler_config = global_scheduler_config
        self.request_queue_config = request_queue_config
        self.replica_scheduler_config = replica_scheduler_config
        self.slo_configs = slo_configs
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.num_pipeline_stages = num_pipeline_stages
        self.num_workers = self.num_tensor_parallel_workers * self.num_pipeline_stages
        self.batch_size = batch_size * num_pipeline_stages

        self.search_config = search_config
        self.start_num_replicas = int(search_config.num_replicas)
        self.start_qps = float(search_config.qps)
        self.qps_multipliers = search_config.qps_multipliers
        self.search_for = search_config.search_for

    def is_valid(self):
        is_model_trace_match = (
            self.model_config.name == self.search_config.model
            and self.trace_config.name == self.search_config.trace
        )
        is_tp_degree_valid = (
            self.model_config.is_tensor_parallel_degree_valid(
                self.num_tensor_parallel_workers
            )
            and self.num_tensor_parallel_workers <= self.cluster_config.gpus_per_node
        )
        # Only allow round_robin global scheduler for single replica qps search
        is_global_scheduler_valid = not (
            self.search_for == "qps"
            and self.start_num_replicas == 1
            and self.global_scheduler_config.scheduler != "round_robin"
        )
        return is_model_trace_match and is_tp_degree_valid and is_global_scheduler_valid

    def get_key(self):
        return (
            f"{self.model_config.get_key()}"
            f"_{self.trace_config.get_key()}"
            f"_{self.cluster_config.get_key()}"
            f"_{self.global_scheduler_config.get_key()}"
            f"_{self.request_queue_config.get_key()}"
            f"_{self.replica_scheduler_config.get_key()}"
            f"_tp{self.num_tensor_parallel_workers}_pp{self.num_pipeline_stages}_bsz{self.batch_size}"
            f"_slo{self.slo_configs.get_key()}"
            f"_search{self.search_config.get_key()}"
        )

    def get_human_readable_name(self):
        return (
            f"Model: {self.model_config.name}"
            f", TP: {self.num_tensor_parallel_workers}, PP: {self.num_pipeline_stages}"
            f", Trace: {self.trace_config.name}"
            f", Device: {self.cluster_config.network_device}"
            f", Global Scheduler: {self.global_scheduler_config.scheduler}"
            f", Request Queue: {self.request_queue_config.name}"
            f", Replica Scheduler: {self.replica_scheduler_config.scheduler}"
            f", BSZ: {self.batch_size}, CS: {self.replica_scheduler_config.chunk_size}"
            f", SLO: {self.slo_configs.name}"
            f", Search: {self.search_config.get_key()}"
            f", Hash: {self.get_hash()}"
        )

    def get_hash(self):
        return hashlib.sha1(self.get_key().encode("utf-8")).hexdigest()[:8]

    def to_config_dict(self):
        return {
            **self.model_config.to_config_dict(),
            **self.cluster_config.to_config_dict(),
            **self.trace_config.to_config_dict(),
            **self.global_scheduler_config.to_config_dict(),
            **self.request_queue_config.to_config_dict(),
            **self.replica_scheduler_config.to_config_dict(),
            "replica_config_tensor_parallel_size": self.num_tensor_parallel_workers,
            "replica_config_num_pipeline_stages": self.num_pipeline_stages,
            "vllm_scheduler_config_batch_size_cap": self.batch_size,
            "orca_scheduler_config_batch_size_cap": self.batch_size,
            "faster_transformer_scheduler_config_batch_size_cap": self.batch_size,
            "sarathi_scheduler_config_batch_size_cap": self.batch_size,
            "vllm_v1_scheduler_config_batch_size_cap": self.batch_size,
        }

    @classmethod
    def generate_job_configs(cls, config: dict):
        job_configs = []
        for (
            search_config,
            model_config,
            cluster_config,
            trace_config,
            global_scheduler_config,
            request_queue_config,
            replica_scheduler_config,
            slo_configs,
            tp_dimension,
            pp_dimension,
            batch_size,
        ) in product(
            config["search_configs"],
            config["models"],
            config["clusters"],
            config["traces"],
            config["global_schedulers"],
            config["request_queues"],
            config["replica_schedulers"],
            config["slo_configs"],
            config["tp_dimensions"],
            config["pp_dimensions"],
            config["batch_sizes"],
        ):
            job_config = cls(
                SearchConfig(**search_config),
                ModelConfig(**model_config),
                ClusterConfig(**cluster_config),
                TraceConfig(**trace_config),
                GlobalSchedulerConfig(**global_scheduler_config),
                RequestQueueConfig(**request_queue_config),
                ReplicaSchedulerConfig(**replica_scheduler_config),
                SloConfigs(**slo_configs),
                tp_dimension,
                pp_dimension,
                batch_size,
            )
            if not job_config.is_valid():
                continue

            job_configs.append(job_config)

        return job_configs

    @classmethod
    def generate_unique_model_job_configs(cls, config: dict, num_requests: int = 32):
        job_configs = []

        trace_config = TraceConfig(**config["traces"][0])
        global_scheduler_config = GlobalSchedulerConfig(
            **config["global_schedulers"][0]
        )
        request_queue_config = RequestQueueConfig(**config["request_queues"][0])
        replica_scheduler_config = ReplicaSchedulerConfig(
            **config["replica_schedulers"][0]
        )
        slo_configs = SloConfigs(**config["slo_configs"][0])
        batch_size = config["batch_sizes"][0]
        # set pp_dimensions to 2 because it covers all the options
        pp_dimensions = [2]

        for model_config, cluster_config, tp_dimension, pp_dimension in product(
            config["models"],
            config["clusters"],
            config["tp_dimensions"],
            pp_dimensions,
        ):
            search_config = SearchConfig(
                trace=trace_config.name,
                model=model_config["name"],
                search_for="qps",
                num_requests=num_requests,
                num_replicas=1,
                qps=1.0,
            )
            job_config = cls(
                search_config,
                ModelConfig(**model_config),
                ClusterConfig(**cluster_config),
                trace_config,
                global_scheduler_config,
                request_queue_config,
                replica_scheduler_config,
                slo_configs,
                tp_dimension,
                pp_dimension,
                batch_size,
            )
            if not job_config.is_valid():
                continue

            job_configs.append(job_config)

        return job_configs


@dataclass
class SimulationConfig:
    output_dir: str
    cache_dir: str
    qps: float
    time_limit: int
    num_replicas: int
    job_config: JobConfig

    def _get_num_requests(self):
        return self.job_config.search_config.num_requests

    def to_config_dict(self):
        return {
            **self.job_config.to_config_dict(),
            "time_limit": self.time_limit * 60,  # to seconds
            "cluster_config_num_replicas": self.num_replicas,
            "metrics_config_output_dir": self.get_run_dir(),
            "interval_generator_config_type": "poisson",
            "poisson_request_interval_generator_config_qps": self.qps,
            "synthetic_request_generator_config_num_requests": self._get_num_requests(),
            "no-metrics_config_save_table_to_wandb": None,
            "no-metrics_config_store_plots": None,
            "no-metrics_config_store_operation_metrics": None,
            "no-metrics_config_store_token_completion_metrics": None,
            "no-metrics_config_keep_individual_batch_metrics": None,
            "no-metrics_config_enable_chrome_trace": None,
            "linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling": None,
            "random_forest_execution_time_predictor_config_skip_cpu_overhead_modeling": None,
            "linear_regression_execution_time_predictor_config_cache_dir": self.cache_dir,
            "random_forest_execution_time_predictor_config_cache_dir": self.cache_dir,
        }

    def to_args(self):
        args = []

        for key, value in self.to_config_dict().items():
            if value is not None:
                args.append(f"--{key} {value}")
            else:
                args.append(f"--{key}")

        return " ".join(args)

    def to_human_readable_name(self):
        return f"{self.job_config.get_human_readable_name()}, Replicas: {self.num_replicas}, QPS: {self.qps}"

    def get_run_dir(self):
        return f"{self.output_dir}/runs/{self.job_config.get_hash()}/r{self.num_replicas}_q{self.qps}"
