from typing import Tuple

from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
    RequestLengthGeneratorOutput,
)


class FixedRequestLengthGenerator(BaseRequestLengthGenerator):
    def get_next_num_tokens(self) -> RequestLengthGeneratorOutput:
        return RequestLengthGeneratorOutput(
            num_prefill_tokens=self._config.prefill_tokens,
            num_decode_tokens=self._config.decode_tokens,
        )
