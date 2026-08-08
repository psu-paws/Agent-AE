"""File to store names for different metrics captured"""

import enum


class OperationMetrics(enum.Enum):
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_DOWN_PROJ_ALL_REDUCE = "mlp_down_proj_all_reduce"
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_POST_PROJ_ALL_REDUCE = "attn_post_proj_all_reduce"
    ATTN_PREFILL = "attn_prefill"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    ATTN_DECODE = "attn_decode"
    ATTN_ROPE = "attn_rope"
    PIPELINE_SEND_RECV = "pipeline_send_recv"
    ADD = "add"
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"


class CpuOperationMetrics(enum.Enum):
    SCHEDULE = "schedule"
    SAMPLER_E2E = "sample_e2e"
    PREPARE_INPUTS_E2E = "prepare_inputs_e2e"
    MODEL_EXECUTION_E2E = "model_execution_e2e"
    PROCESS_MODEL_OUTPUTS = "process_model_outputs"
    RAY_COMM_TIME = "ray_comm_time"


class RequestMetricsTimeDistributions(enum.Enum):
    REQUEST_E2E_TIME = "request_e2e_time"
    REQUEST_E2E_TIME_NORMALIZED = "request_e2e_time_normalized"
    REQUEST_EXECUTION_TIME = "request_execution_time"
    REQUEST_EXECUTION_TIME_NORMALIZED = "request_execution_time_normalized"
    REQUEST_MODEL_EXECUTION_TIME = "request_model_execution_time"
    REQUEST_MODEL_EXECUTION_TIME_NORMALIZED = "request_model_execution_time_normalized"
    REQUEST_PREEMPTION_TIME = "request_preemption_time"
    REQUEST_SCHEDULING_DELAY = "request_scheduling_delay"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME = "request_execution_plus_preemption_time"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED = (
        "request_execution_plus_preemption_time_normalized"
    )
    PREFILL_TIME_E2E = "prefill_e2e_time"
    PREFILL_TIME_E2E_NORMALIZED = "prefill_e2e_time_normalized"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION = "prefill_time_execution_plus_preemption"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "prefill_time_execution_plus_preemption_normalized"
    )
    DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "decode_time_execution_plus_preemption_normalized"
    )


class RequestMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_PREFILL_TOKENS_CACHED = "request_num_prefill_tokens_cached"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_NUM_RESTARTS = "request_num_restarts"
    REQUEST_ARRIVED_AT = "request_arrived_at"


class RequestMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_NUM_PREFILL_TOKENS = "batch_num_prefill_tokens"
    BATCH_NUM_DECODE_TOKENS = "batch_num_decode_tokens"
    BATCH_SIZE = "batch_size"
    BATCH_NUM_COMPLETED_PREFILLS = "batch_num_completed_prefills"
    # BATCH_DECODE_CONTEXT_SUM = "batch_decode_context_sum"
    # BATCH_DECODE_CONTEXT_SPREAD = "batch_decode_context_spread"
    # BATCH_DECODE_CONTEXT_IQR = "batch_decode_context_iqr"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"


class BatchMetricsTimeSeries(enum.Enum):
    BATCH_DECODE_CONTEXT_SUM_TIMESERIES = "batch_decode_context_sum_timeseries"
    BATCH_DECODE_CONTEXT_SPREAD_TIMESERIES = "batch_decode_context_spread_timeseries"
    BATCH_DECODE_CONTEXT_IQR_TIMESERIES = "batch_decode_context_iqr_timeseries"
    BATCH_NUM_PREFILL_TOKENS_TIMESERIES = "batch_num_prefill_tokens_timeseries"


class TokenMetricsTimeDistribution(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME = (
        "decode_token_execution_plus_preemption_time"
    )


class TokenMetricsTimeSeries(enum.Enum):
    PREFILL_ARRIVAL = "prefill_arrival"
    PREFILL_COMPLETIONS = "prefill_completion"
    PREFILL_TOKENS_OUTSTANDING = "prefill_tokens_outstanding"
    DECODE_COMPLETIONS = "decode_completion"
