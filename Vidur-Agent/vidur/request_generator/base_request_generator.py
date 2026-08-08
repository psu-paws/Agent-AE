from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from vidur.config import BaseRequestGeneratorConfig
from vidur.entities import Request


class BaseRequestGenerator(ABC):
    def __init__(self, config: BaseRequestGeneratorConfig):
        self._config = config
        self._random_number_generator = np.random.default_rng(config.seed)

    @abstractmethod
    def get_next_request_arrival_time(self, idx) -> Optional[float]:
        pass

    @abstractmethod
    def get_next_request(self) -> Optional[Request]:
        pass
