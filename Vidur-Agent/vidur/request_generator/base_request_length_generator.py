from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from vidur.config import BaseRequestLengthGeneratorConfig


class RequestLengthGeneratorOutput:
    num_prefill_tokens: int
    num_decode_tokens: int
    block_hash_ids: Optional[List[int]]
    block_size: Optional[int]
    session_id: Optional[int]
    turn_id: Optional[int]
    model_id: Optional[int]
    request_id: Optional[int]

    def __init__(
        self,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        block_hash_ids: Optional[List[int]],
        token_ids: Optional[List[int]],
        block_size: Optional[int],
        session_id: Optional[int],
        model_id: Optional[int],
        request_id: Optional[int],
        turn_id: Optional[int] = None,
        inter_request_latency: float = 0.0,
    ):
        self.num_prefill_tokens = int(num_prefill_tokens)
        self.num_decode_tokens = int(num_decode_tokens)
        self.block_hash_ids = block_hash_ids
        self.token_ids = token_ids
        self.inter_request_latency = float(inter_request_latency)
        self.block_size = block_size
        self.session_id = session_id
        self.turn_id = turn_id
        self.model_id = model_id
        self.request_id = request_id


class BaseRequestLengthGenerator(ABC):
    def __init__(
        self,
        config: BaseRequestLengthGeneratorConfig,
        random_number_generator: Generator,
    ):
        self._config = config
        self._random_number_generator = random_number_generator

    @abstractmethod
    def get_next_num_tokens(self) -> RequestLengthGeneratorOutput:
        pass
