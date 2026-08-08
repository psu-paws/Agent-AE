from typing import Generator

from vidur.config import ZipfRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
    RequestLengthGeneratorOutput,
)
from vidur.utils.zipf_generator import ZipfGenerator


class ZipfRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(
        self,
        config: ZipfRequestLengthGeneratorConfig,
        random_number_generator: Generator,
    ):
        super().__init__(config, random_number_generator)

        self.zipf_generator = ZipfGenerator(
            config.min_tokens,
            config.max_tokens,
            config.theta,
            config.scramble,
            config.seed,
        )

    def get_next_num_tokens(self) -> RequestLengthGeneratorOutput:
        total_tokens = self.zipf_generator.next()

        decode_tokens = total_tokens / (1 + self._config.prefill_to_decode_ratio)
        prefill_tokens = total_tokens - decode_tokens

        return RequestLengthGeneratorOutput(
            num_prefill_tokens=prefill_tokens,
            num_decode_tokens=decode_tokens,
        )
