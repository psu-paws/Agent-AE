from vidur.config import CacheConfig, ReplicaConfig
from vidur.utils.param_counter import ParamCounter


class MemoryPlanner:
    def __init__(
        self, replica_config: ReplicaConfig, cache_config: CacheConfig
    ) -> None:
        self._replica_config = replica_config
        self._device_config = replica_config.device_config
        self._cache_config = cache_config
        self._param_counter = ParamCounter(replica_config)

    def _get_parameter_memory_per_device(self) -> int:
        return (
            2 * self._param_counter.get_num_parameters_per_device()
        )  # 2 bytes per float

    def _get_kv_cache_memory_per_layer_per_token(self) -> int:
        return (
            2  # 2 bytes per float
            * 2  # one for key, one for value
            * self._replica_config.attention_head_dim
            * self._replica_config.kv_heads_per_tensor_parallel_worker
            * 1  # one token
        )

    def _get_kv_cache_memory_per_device_per_token(self) -> int:
        return (
            self._get_kv_cache_memory_per_layer_per_token()
            * self._replica_config.num_layers_per_pipeline_stage
        )

    def get_max_kv_cache_size_in_tokens(self) -> int:
        available_memory = (
            self._device_config.total_memory_gb
            * 1024**3
            * (1 - self._cache_config.memory_margin_fraction)
        )
        parameter_memory_per_device = self._get_parameter_memory_per_device()
        kv_cache_memory_per_device_per_token = (
            self._get_kv_cache_memory_per_device_per_token()
        )

        memory_for_kv_cache = available_memory - parameter_memory_per_device
        number_of_tokens = memory_for_kv_cache // kv_cache_memory_per_device_per_token

        assert number_of_tokens > 0, "Not enough memory to store even a single token"

        return number_of_tokens
