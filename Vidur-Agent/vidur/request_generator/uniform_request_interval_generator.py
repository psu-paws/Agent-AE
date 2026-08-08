from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class UniformRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._qps = self._config.qps

    def get_next_inter_request_time(self) -> float:
        return 1 / self._qps
