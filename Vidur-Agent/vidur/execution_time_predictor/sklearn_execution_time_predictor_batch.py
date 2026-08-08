from math import ceil
from typing import List

import numpy as np

from vidur.entities.batch import Batch
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)


class SklearnExecutionTimePredictorBatch:
    def __init__(
        self,
        execution_time_predictor_requests: List[ExecutionTimePredictorRequest],
        kv_cache_prediction_granularity: int,
        prefill_chunk_size_prediction_granularity: int,
    ):
        self._kv_cache_prediction_granularity = kv_cache_prediction_granularity
        self._prefill_chunk_size_prediction_granularity = (
            prefill_chunk_size_prediction_granularity
        )
        self._total_num_tokens = sum(
            req.num_tokens_to_process for req in execution_time_predictor_requests
        )
        self._total_num_tokens_rounded = ((self._total_num_tokens + 7) // 8) * 8
        self._size = len(execution_time_predictor_requests)
        self._prefill_batch_size = sum(
            1
            for req in execution_time_predictor_requests
            if not req.is_prefill_complete
        )
        self._decode_batch_size = self._size - self._prefill_batch_size

        if self._decode_batch_size > 0:
            decode_kv_cache_sizes = [
                request.num_processed_tokens
                for request in execution_time_predictor_requests
                if request.is_prefill_complete
            ]
            self._decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
            self._decode_avg_kv_cache_size = (
                (
                    self._decode_avg_kv_cache_size
                    + self._kv_cache_prediction_granularity
                    - 1
                )
                // self._kv_cache_prediction_granularity
            ) * self._kv_cache_prediction_granularity

        if self._prefill_batch_size > 0:
            self._prefill_agg_kv_cache_size = sum(
                req.num_processed_tokens
                for req in execution_time_predictor_requests
                if not req.is_prefill_complete
            )
            self._prefill_agg_kv_cache_size = (
                (
                    self._prefill_agg_kv_cache_size
                    + self._kv_cache_prediction_granularity
                    - 1
                )
                // self._kv_cache_prediction_granularity
            ) * self._kv_cache_prediction_granularity
            self._prefill_agg_chunk_size = round(
                sum(
                    req.num_tokens_to_process**2
                    for req in execution_time_predictor_requests
                    if not req.is_prefill_complete
                )
                ** 0.5
            )
            self._prefill_agg_chunk_size = (
                ceil(
                    self._prefill_agg_chunk_size
                    / self._prefill_chunk_size_prediction_granularity
                )
            ) * self._prefill_chunk_size_prediction_granularity

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def total_num_tokens_rounded(self) -> int:
        return self._total_num_tokens_rounded

    @property
    def size(self) -> int:
        return self._size

    @property
    def decode_batch_size(self) -> int:
        return self._decode_batch_size

    @property
    def decode_avg_kv_cache_size(self) -> int:
        return self._decode_avg_kv_cache_size

    @property
    def prefill_batch_size(self) -> int:
        return self._prefill_batch_size

    @property
    def prefill_agg_kv_cache_size(self) -> int:
        return self._prefill_agg_kv_cache_size

    @property
    def prefill_agg_chunk_size(self) -> int:
        return self._prefill_agg_chunk_size

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        kv_cache_prediction_granularity: int,
        prefill_chunk_size_prediction_granularity: int,
    ):
        return cls(
            [
                ExecutionTimePredictorRequest(
                    req.num_processed_tokens,
                    num_tokens,
                    req.is_prefill_complete,
                )
                for req, num_tokens in zip(batch.requests, batch.num_tokens)
            ],
            kv_cache_prediction_granularity,
            prefill_chunk_size_prediction_granularity,
        )
