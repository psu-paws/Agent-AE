import json
import logging
from typing import Generator

import numpy as np
import pandas as pd

from vidur.config import TraceRequestLengthGeneratorConfig
from vidur.request_generator.session_utils import resolve_session_and_turn
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
    RequestLengthGeneratorOutput,
)

logger = logging.getLogger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(
        self,
        config: TraceRequestLengthGeneratorConfig,
        random_number_generator: Generator,
    ):
        super().__init__(config, random_number_generator)

        self.trace_df = pd.read_csv(config.trace_file)

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

        # assert that the total number of tokens does not exceed the max tokens
        assert (config.max_tokens is None) or all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )
        assert all(self.trace_df["num_prefill_tokens"] > 0)
        # assert all(self.trace_df["num_decode_tokens"] > 0)

        if "token_ids" in self.trace_df.columns:
            self.trace_df["token_ids"] = self.trace_df["token_ids"].apply(
                json.loads
            )
        else:
            self.trace_df["token_ids"] = None

        # Preprocess block_hash_ids, block_size
        if "block_hash_ids" in self.trace_df.columns:
            self.trace_df["block_hash_ids"] = self.trace_df["block_hash_ids"].apply(
                json.loads
            )
        else:
            self.trace_df["block_hash_ids"] = None

        if "block_size" not in self.trace_df.columns:
            self.trace_df["block_size"] = None

        # Derive session_id / turn_id from whatever columns the trace carries.
        self.trace_df = resolve_session_and_turn(self.trace_df)
        
        # Shim model_id to None if not present
        if "model_id" not in self.trace_df.columns:
            self.trace_df["model_id"] = None
        else:
            self.trace_df["model_id"] = self.trace_df["model_id"].astype(int)
        if "request_id" not in self.trace_df.columns:
            self.trace_df["request_id"] = None
        else:
            self.trace_df["request_id"] = self.trace_df["request_id"].astype(int)
        # num_sessions = number of sessions; a session groups all turns of one
        # conversation, and is the unit of arrival chaining and dependency release.
        self.num_sessions = self.trace_df["session_id"].nunique()

        if "inter_request_latency" not in self.trace_df.columns:
            self.trace_df["inter_request_latency"] = 0.0
        else:
            self.trace_df["inter_request_latency"] = self.trace_df["inter_request_latency"].astype(float).fillna(0.0)
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles.
        # Requests with no decode tokens are dropped from the summary.
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        ).replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(
            f"Loaded request length trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        pd_distribution = pd_ratio.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        logger.debug(f"Prompt/decode token ratio stats\n: {pd_distribution}")

        # randomly shuffle the df based on the seed
        if not self._config.preserve_request_order:
            self.trace_df = self.trace_df.sample(frac=1, random_state=self._config.seed)
        
        self.session_row_indices = [(self.trace_df.index[self.trace_df['session_id'] == i].tolist()) for i in range(self.num_sessions)]
        self.next_request_idx = [0 for _ in range(self.num_sessions)]

        # dep column: list of turn_ids within the same session that must ALL
        # complete before this request is enqueued.
        if "dep" in self.trace_df.columns:
            self.trace_df["dep"] = self.trace_df["dep"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            )
            self.has_deps = True
        else:
            self.trace_df["dep"] = [[] for _ in range(len(self.trace_df))]
            self.has_deps = False

        # Build per-session lookup structures for dep-based scheduling.
        # turn_to_pos[session_id][turn_id] = position in session_row_indices[session_id]
        # dep_children[session_id][dep_turn_id] = [child_turn_ids]
        self.turn_to_pos = {}
        self.dep_children = {}
        for session_id in range(self.num_sessions):
            self.turn_to_pos[session_id] = {}
            self.dep_children[session_id] = {}
            for pos, row_idx in enumerate(self.session_row_indices[session_id]):
                row = self.trace_df.iloc[row_idx]
                turn_id = int(row["turn_id"])
                self.turn_to_pos[session_id][turn_id] = pos
                for dep_turn in row["dep"]:
                    self.dep_children[session_id].setdefault(int(dep_turn), []).append(turn_id)

    def get_num_sessions(self) -> int:
        return self.num_sessions

    def get_num_tokens_for_turn(self, session_id, turn_id) -> RequestLengthGeneratorOutput:
        pos = self.turn_to_pos[session_id][turn_id]
        row = self.trace_df.iloc[self.session_row_indices[session_id][pos]]
        return RequestLengthGeneratorOutput(
            num_prefill_tokens=row["num_prefill_tokens"],
            num_decode_tokens=row["num_decode_tokens"],
            block_hash_ids=row["block_hash_ids"],
            token_ids=row["token_ids"],
            block_size=row["block_size"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            model_id=row["model_id"],
            request_id=row["request_id"],
            inter_request_latency=row["inter_request_latency"],
        )

    def get_next_num_tokens(self, session_id) -> RequestLengthGeneratorOutput:
        # Bounded by the caller: _generate_next_request stops once a session has
        # issued all of its turns, so the cursor never runs past its session.
        assert self.next_request_idx[session_id] < len(self.session_row_indices[session_id])

        row = self.trace_df.iloc[self.session_row_indices[session_id][self.next_request_idx[session_id]]]
        self.next_request_idx[session_id] += 1

        return RequestLengthGeneratorOutput(
            num_prefill_tokens=row["num_prefill_tokens"],
            num_decode_tokens=row["num_decode_tokens"],
            block_hash_ids=row["block_hash_ids"],
            token_ids=row["token_ids"],
            block_size=row["block_size"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            model_id=row["model_id"],
            request_id=row["request_id"],
            inter_request_latency=row["inter_request_latency"],
        )
