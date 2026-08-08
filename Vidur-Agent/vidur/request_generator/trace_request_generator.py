import json
import logging
from typing import Optional

import pandas as pd

from vidur.config import TraceRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        self.num_requests_generated = 0
        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)
        # assert that the number of requests is not greater than the number of requests in the trace
        self.num_requests = (
            config.num_requests
            if config.num_requests is not None
            else len(self.trace_df)
        )
        assert self.num_requests <= len(self.trace_df)

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # assert that the total number of tokens does not exceed the max tokens
        assert (config.max_tokens is None) or all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )

        if "model_id" not in self.trace_df.columns:
            self.trace_df["model_id"] = None
        else:
            self.trace_df["model_id"] = self.trace_df["model_id"].astype(int)


        if "request_id" not in self.trace_df.columns:
            self.trace_df["request_id"] = None
        else:
            self.trace_df["request_id"] = self.trace_df["request_id"].astype(int)

        # rescale the time to change QPS
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )

        # Preprocess block_hash_ids, block_size
        if "block_hash_ids" in self.trace_df.columns:
            self.trace_df["block_hash_ids"] = self.trace_df["block_hash_ids"].apply(
                json.loads
            )
        else:
            self.trace_df["block_hash_ids"] = None

        if "token_ids" in self.trace_df.columns:
            self.trace_df["token_ids"] = self.trace_df["token_ids"].apply(
                json.loads
            )
        else:
            self.trace_df["token_ids"] = None

        if "block_size" not in self.trace_df.columns:
            self.trace_df["block_size"] = None

        # Shim session_id to None if not present
        if "session_id" not in self.trace_df.columns:
            self.trace_df["session_id"] = None
        else:
            self.trace_df["session_id"] = self.trace_df["session_id"].astype(int)

        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def get_next_request_arrival_time(self) -> Optional[float]:
        if self.num_requests_generated >= self.num_requests:
            return None

        return self.trace_df.iloc[self.num_requests_generated]["arrived_at"]

    def get_next_request(self) -> Optional[Request]:
        if self.num_requests_generated >= self.num_requests:
            return None

        request_from_trace = self.trace_df.iloc[self.num_requests_generated]
        request = Request(
            arrived_at=request_from_trace["arrived_at"],
            num_prefill_tokens=request_from_trace["num_prefill_tokens"],
            num_decode_tokens=request_from_trace["num_decode_tokens"],
            block_hash_ids=request_from_trace["block_hash_ids"],
            token_ids=request_from_trace["token_ids"],
            block_size=request_from_trace["block_size"],
            session_id=request_from_trace["session_id"],
            model_id=request_from_trace["model_id"],
            request_id=request_from_trace["request_id"],
        )
        self.num_requests_generated += 1
        return request
