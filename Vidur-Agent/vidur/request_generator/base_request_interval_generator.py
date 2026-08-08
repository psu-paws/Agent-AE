from abc import ABC, abstractmethod
from typing import Generator

from vidur.config import BaseRequestIntervalGeneratorConfig


class BaseRequestIntervalGenerator(ABC):
    def __init__(
        self,
        config: BaseRequestIntervalGeneratorConfig,
        random_number_generator: Generator,
    ):
        self._config = config
        self._random_number_generator = random_number_generator

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass
