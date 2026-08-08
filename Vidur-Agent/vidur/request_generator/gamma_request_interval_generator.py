from typing import Generator

from scipy.stats import gamma

from vidur.config import GammaRequestIntervalGeneratorConfig
from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class GammaRequestIntervalGenerator(BaseRequestIntervalGenerator):

    def __init__(
        self,
        config: GammaRequestIntervalGeneratorConfig,
        random_number_generator: Generator,
    ):
        super().__init__(config, random_number_generator)

        cv = self._config.cv
        self.qps = self._config.qps
        self.gamma_shape = 1.0 / (cv**2)

    def get_next_inter_request_time(self) -> float:
        gamma_scale = 1.0 / (self.qps * self.gamma_shape)
        return gamma.rvs(
            self.gamma_shape,
            scale=gamma_scale,
            random_state=self._random_number_generator,
        )
