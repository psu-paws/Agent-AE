from typing import Dict, List, Optional

import pandas as pd

from vidur.config import SimulationConfig
from vidur.entities import Batch, BatchStage, ExecutionTime, Request
from vidur.logger import init_logger
from vidur.metrics.data_structures.cdf_sketch import CDFSketch
from vidur.metrics.data_structures.data_series import DataSeries
from vidur.metrics.metrics import (
    CpuOperationMetrics,
    OperationMetrics,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    RequestMetricsTimeSeries,
    TokenMetricsTimeDistribution,
    TokenMetricsTimeSeries,
)

logger = init_logger(__name__)


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
BATCH_ID_STR = "Batch Id"
MEMORY_USAGE_STR = "Memory Usage (%)"
BUSY_TIME_PERCENT = "Busy Time (%)"
UTILIZATION_STR = "Utilization (%)"
OPERATION_STR = "Operation"
TIME_STR_MS = "Time (ms)"


class RequestMetricsStore:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        request_id: Optional[int] = None,
        replica_config=None,
    ) -> None:
        self._request_id = request_id
        self._simulation_config = simulation_config

        # copy config
        self._config = self._simulation_config.metrics_config
        rc = replica_config or self._simulation_config.cluster_config.replica_config
        self._num_pipeline_stages = rc.num_pipeline_stages
        self._last_request_arrived_at = 0.0

        # Initialise request metrics
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._request_metrics_time_series: Dict[
            RequestMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeSeries:
            self._request_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )


        # Initialise token metrics
        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._token_metrics_time_series: Dict[TokenMetricsTimeSeries, DataSeries] = {}
        for metric_name in TokenMetricsTimeSeries:
            self._token_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise operation metrics
        self._operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        self._operation_metrics_per_batch: Dict[OperationMetrics, DataSeries] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, CDFSketch] = {}
        self._cpu_operation_metrics_per_batch: Dict[CpuOperationMetrics, DataSeries] = (
            {}
        )
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._cpu_operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

    @if_write_metrics
    def on_request_arrival(self, request: Request) -> None:
        if not self._config.store_request_metrics:
            return
        self._request_metrics_time_series[RequestMetricsTimeSeries.REQUEST_ARRIVAL].put(
            request.arrived_at, 1
        )
        self._token_metrics_time_series[TokenMetricsTimeSeries.PREFILL_ARRIVAL].put(
            request.arrived_at, request.num_prefill_tokens
        )
        self._token_metrics_time_series[
            TokenMetricsTimeSeries.PREFILL_TOKENS_OUTSTANDING
        ].put(request.arrived_at, request.num_prefill_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(request.id, request.num_prefill_tokens)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(request.id, request.num_decode_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
        ].put(request.id, request.arrived_at - self._last_request_arrived_at)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_ARRIVED_AT].put(
            request.id, request.arrived_at
        )
        self._last_request_arrived_at = request.arrived_at


    @if_write_metrics
    def on_request_end(self, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        self._request_metrics_time_series[
            RequestMetricsTimeSeries.REQUEST_COMPLETION
        ].put(request.completed_at, 1)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.completed_at - request.arrived_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, (request.completed_at - request.arrived_at) / request.num_decode_tokens if request.num_decode_tokens else 0.0)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            request.id,
            (request.execution_time + request.preempted_time)
            / request.num_decode_tokens if request.num_decode_tokens else 0.0,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.arrived_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens if request.num_decode_tokens else 0.0,
        )

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS_CACHED
        ].put(request.id, request.num_prefill_tokens_cached)

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        if not self._config.store_token_completion_metrics:
            return

        if request.has_started_decode:
            self._token_metrics_time_distribution[
                TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME
            ].put(
                time - batch.scheduled_at + request.latest_iteration_scheduling_delay,
            )
            # TODO(t-nitinkedia): Enable decode completions behind a feature flag because of perf overhead
            # self._token_metrics_time_series[
            #     TokenMetricsTimeSeries.DECODE_COMPLETIONS
            # ].put(time, 1)
        elif time == request.prefill_completed_at:
            # if prefill has just finished in this iteration, update the prefill completion time series
            self._token_metrics_time_series[
                TokenMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                time,
                request.num_prefill_tokens,
            )

    def _push_metric(
        self, metric_name: OperationMetrics, batch_id: int, value: float
    ) -> None:
        if metric_name in OperationMetrics:
            self._operation_metrics[metric_name].put(value)
            if self._config.keep_individual_batch_metrics:
                self._operation_metrics_per_batch[metric_name].put(batch_id, value)
        elif metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name].put(value)
            if self._config.keep_individual_batch_metrics:
                self._cpu_operation_metrics_per_batch[metric_name].put(batch_id, value)
        else:
            raise ValueError(f"Invalid metric name {metric_name}")

    @if_write_metrics
    def on_replica_schedule(self, time: float, memory_usage_percent: int) -> None:
        if not self._config.store_utilization_metrics:
            return

    @if_write_metrics
    def on_replica_stage_schedule(
        self,
        time: float,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> None:

        if not self._config.store_operation_metrics:
            return

        batch_id = batch_stage._batch_id
        self._push_metric(
            OperationMetrics.MLP_UP_PROJ,
            batch_id,
            execution_time.mlp_layer_up_proj_execution_time,
        )
        self._push_metric(
            OperationMetrics.MLP_ACTIVATION,
            batch_id,
            execution_time.mlp_layer_act_execution_time,
        )
        self._push_metric(
            OperationMetrics.MLP_DOWN_PROJ,
            batch_id,
            execution_time.mlp_layer_down_proj_execution_time,
        )
        self._push_metric(
            OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            batch_id,
            execution_time.mlp_all_reduce_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_PRE_PROJ,
            batch_id,
            execution_time.attention_pre_proj_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_POST_PROJ,
            batch_id,
            execution_time.attention_post_proj_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            batch_id,
            execution_time.attention_all_reduce_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_PREFILL,
            batch_id,
            execution_time.attention_prefill_execution_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_DECODE,
            batch_id,
            execution_time.attention_decode_execution_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_KV_CACHE_SAVE,
            batch_id,
            execution_time.attention_kv_cache_save_execution_time,
        )
        self._push_metric(
            OperationMetrics.ATTN_ROPE,
            batch_id,
            execution_time.attention_rope_execution_time,
        )
        self._push_metric(OperationMetrics.ADD, batch_id, execution_time.add_time * 2)
        self._push_metric(
            OperationMetrics.INPUT_LAYERNORM,
            batch_id,
            execution_time.attn_norm_time,
        )
        self._push_metric(
            OperationMetrics.POST_ATTENTION_LAYERNORM,
            batch_id,
            execution_time.mlp_norm_time,
        )

        self._push_metric(
            OperationMetrics.PIPELINE_SEND_RECV,
            batch_id,
            execution_time.pipeline_parallel_communication_time,
        )
        self._push_metric(
            CpuOperationMetrics.SCHEDULE, batch_id, execution_time.schedule_time
        )
        self._push_metric(
            CpuOperationMetrics.SAMPLER_E2E, batch_id, execution_time.sampler_e2e_time
        )
        self._push_metric(
            CpuOperationMetrics.PREPARE_INPUTS_E2E,
            batch_id,
            execution_time.prepare_inputs_e2e_time,
        )
        self._push_metric(
            CpuOperationMetrics.MODEL_EXECUTION_E2E,
            batch_id,
            execution_time.model_time_ms,
        )
        self._push_metric(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS,
            batch_id,
            execution_time.process_model_outputs_time,
        )
        self._push_metric(
            CpuOperationMetrics.RAY_COMM_TIME, batch_id, execution_time.ray_comm_time
        )

    @if_write_metrics
    def on_batch_stage_end(
        self, batch_stage: BatchStage, time: float, stage_id: int
    ) -> None:
        if not self._config.store_utilization_metrics:
            return

    @if_write_metrics
    def on_batch_end(self, time: float, request: Request, batch: Batch) -> None:
        """Record per-token execution times for this store's own request.

        Takes a single request rather than the whole batch: the caller invokes
        this once per request in the batch, so iterating the batch here would do
        O(batch^2) work and would record every session's token times in every
        other session's store.

        Batch-level metrics are deliberately not collected here. They are
        cluster-scoped and already recorded once per batch by
        ReplicaMetricsStore; duplicating them in every session's store cost one
        sketch update per metric per request and produced *_request.csv files
        that nothing consumes.
        """
        self._update_per_token_execution_times(time, request, batch)

        if not self._config.store_batch_metrics:
            return

        self._token_metrics_time_series[
            TokenMetricsTimeSeries.PREFILL_TOKENS_OUTSTANDING
        ].put(time, -1 * batch.num_prefill_tokens)

    def store_metrics(self, base_plot_path: str, sim_time: float):
        if self._config.store_request_metrics:
            for dataseries in self._request_metrics_histogram.values():
                dataseries.plot_histogram(
                    base_plot_path, dataseries._y_name, bin_count=25
                )

            for dataseries in self._request_metrics_time_distributions.values():
                dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)
                dataseries.plot_scatter(
                    base_plot_path, f"{dataseries._y_name}_scatter", TIME_STR
                )

            for dataseries in self._request_metrics_time_series.values():
                dataseries.plot_step(
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                )

        if self._config.store_token_completion_metrics:
            for dataseries in self._token_metrics_time_series.values():
                dataseries.plot_step(
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                )

    def get_merged_df(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
    ):
        dfs = [dataseries.to_df() for dataseries in dataseries_list]
        assert all([df[key_to_join].is_unique for df in dfs])
        # assert all([len(df) == len(dfs[0]) for df in dfs])

        # https://stackoverflow.com/questions/53645882/pandas-merging-101/65167327#65167327
        merged_df = pd.concat(
            [df.set_index(key_to_join) for df in dfs], axis=1, join="inner"
        ).reset_index()

        merged_df["request"] = self._request_id
        return merged_df

    def get_request_metrics_df(self):
        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())
        return self.get_merged_df(all_request_metrics, REQUEST_ID_STR)

    def get_operation_metrics_df(self):
        all_operation_metrics = list(self._operation_metrics_per_batch.values()) + list(
            self._cpu_operation_metrics_per_batch.values()
        )
        return self.get_merged_df(all_operation_metrics, BATCH_ID_STR)

    def get_replica_busy_time(self):
        replica_busy_time_dict = {}
        for stage_idx in range(self._num_pipeline_stages):
            replica_busy_time_dict[stage_idx] = self._replica_busy_time[
                stage_idx
            ].get_stats("replica_busy_time")
        return replica_busy_time_dict

    def get_replica_mfu(self):
        replica_mfu_dict = {}
        for stage_idx in range(self._num_pipeline_stages):
            replica_mfu_dict[stage_idx] = self._replica_mfu[stage_idx].get_stats(
                "replica_mfu"
            )
        return replica_mfu_dict

    def get_prefix_cache_stats(self):
        cached_tokens_stats = self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS_CACHED
        ].get_stats()
        if cached_tokens_stats is None:
            return {}
        total_tokens_stats = self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].get_stats()
        if total_tokens_stats is None:
            return {}
        return {
            "cached_tokens_sum": int(cached_tokens_stats["sum"]),
            "total_tokens_sum": int(total_tokens_stats["sum"]),
            "hit_ratio": cached_tokens_stats["sum"] / total_tokens_stats["sum"],
        }
