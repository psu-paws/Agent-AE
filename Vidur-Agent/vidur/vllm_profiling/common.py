"""Timing, model config and sweep spaces shared by the profilers.

Both the timing method and the sweep spaces follow vidur/profiling, so the
resulting CSVs are drop-in compatible with the execution-time predictor.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

WARMUP_STEPS = 2
ACTIVE_STEPS = 5


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
class TimerStatsStore:
    """Collects CUDA-event pairs per op name and reduces them to stats in ms."""

    def __init__(self, disabled: bool = False):
        self.disabled = disabled
        self.TIMING_STATS: Dict[str, list] = {}

    def record_time(self, name: str, event_pair):
        self.TIMING_STATS.setdefault(name, []).append(event_pair)

    def clear_stats(self):
        self.TIMING_STATS = {}

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, times in self.TIMING_STATS.items():
            elapsed = [
                (t if isinstance(t, float) else t[0].elapsed_time(t[1])) for t in times
            ]
            stats[name] = {
                "min": float(np.min(elapsed)),
                "max": float(np.max(elapsed)),
                "mean": float(np.mean(elapsed)),
                "median": float(np.median(elapsed)),
                "std": float(np.std(elapsed)),
            }
        return stats


class CudaTimer:
    """Context manager recording one CUDA-event pair per __enter__/__exit__."""

    def __init__(self, name: Optional[str], store: TimerStatsStore):
        self.name = name
        self.store = store
        self.disabled = name is None or store.disabled
        self.start_event = None

    def __enter__(self):
        if self.disabled:
            return self
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        if self.disabled:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self.store.record_time(self.name, [self.start_event, end_event])


# --------------------------------------------------------------------------
# model config, read from the HF config so it matches what vLLM serves
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model: str
    embedding_dim: int
    mlp_hidden_dim: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False

    @property
    def head_size(self) -> int:
        return self.embedding_dim // self.num_q_heads

    @classmethod
    def from_model_name(cls, model: str) -> "ModelConfig":
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(model, trust_remote_code=True)
        # Some architectures use QKV bias but no bias elsewhere.
        qkv_bias = bool(getattr(c, "attention_bias", False) or False)
        if getattr(c, "model_type", "") in ("qwen2", "qwen2_moe"):
            qkv_bias = True
        return cls(
            model=model,
            embedding_dim=c.hidden_size,
            mlp_hidden_dim=c.intermediate_size,
            num_q_heads=c.num_attention_heads,
            num_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
            num_layers=c.num_hidden_layers,
            vocab_size=c.vocab_size,
            rope_theta=float(getattr(c, "rope_theta", 10000.0)),
            rms_norm_eps=float(getattr(c, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(getattr(c, "max_position_embeddings", 32768)),
            use_gated_mlp=str(getattr(c, "hidden_act", "silu")).startswith("silu"),
            use_bias=False,
            use_qkv_bias=qkv_bias,
        )

    def num_q_heads_per_worker(self, tp: int) -> int:
        assert self.num_q_heads % tp == 0, f"{self.num_q_heads} q heads not divisible by tp={tp}"
        return self.num_q_heads // tp

    def num_kv_heads_per_worker(self, tp: int) -> int:
        # vLLM replicates KV heads when tp > num_kv_heads (GQA); mirror that.
        return max(1, self.num_kv_heads // tp)

    def mlp_hidden_per_worker(self, tp: int) -> int:
        assert self.mlp_hidden_dim % tp == 0
        return self.mlp_hidden_dim // tp


# --------------------------------------------------------------------------
# sweep spaces, from vidur/profiling/utils
# --------------------------------------------------------------------------
def get_num_tokens_to_profile(max_num_tokens: int) -> List[int]:
    space = (
        [1, 2, 4]
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, 4 * 1024 + 1, 32))
        + list(range(4 * 1024, 8 * 1024 + 1, 64))
        + list(range(8 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 32 * 1024 + 1, 256))
        + list(range(32 * 1024, 64 * 1024 + 1, 512))
        + list(range(64 * 1024, 128 * 1024 + 1, 1024))
    )
    out = [n for n in space if n <= max_num_tokens]
    out.sort(reverse=True)
    return out


def get_attention_batch_sizes_to_profile(min_bs: int, max_bs: int) -> List[int]:
    space = list(range(1, 128 + 1, 1)) + list(range(128, 1024 + 1, 8))
    return [x for x in space if min_bs <= x <= max_bs]


def get_attention_prefill_chunk_sizes_to_profile(max_seq_len, max_chunk) -> List[int]:
    space = (
        list(range(32, 128 + 1, 32))
        + list(range(256, 1024 + 1, 128))
        + list(range(1024, 4 * 1024 + 1, 512))
        + list(range(4 * 1024, 16 * 1024 + 1, 512))
        + list(range(16 * 1024, 64 * 1024 + 1, 1024))
        + list(range(64 * 1024, 128 * 1024 + 1, 2048))
    )
    out = []
    for c in space:
        if c <= max_seq_len and c <= max_chunk:
            out.append(c)
        else:
            break
    return out


def get_seq_lengths_to_profile(max_seq_len: int) -> List[int]:
    # Tree-based predictors return their boundary value outside the training
    # range, so the sweep must reach the longest context the workload uses.
    # The last band widens to whatever the caller asks for.
    space = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 64 * 1024 + 1, 256))
        + list(range(64 * 1024, max(max_seq_len, 64 * 1024) + 1, 1024))
    )
    return [s for s in sorted(set(space)) if s < max_seq_len]


def get_collectives_sizes_to_profile(max_size: int) -> List[int]:
    MB, GB = 1024 * 1024, 1024 * 1024 * 1024
    space = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))
        + list(range(512 * 1024, 8 * MB + 1, 16 * 1024))
        + list(range(8 * MB, 64 * MB + 1, 64 * 1024))
        # 265 (not 256) is vidur's own step; kept so the rows line up.
        + list(range(64 * MB + 1, 512 * MB + 1, 265 * 1024))
        # A prefill/decode KV handoff moves several GB. Steps are coarse because
        # the curve is linear in size well before that point, so few samples are
        # needed to pin the slope. The last band follows the caller's maximum.
        + list(range(512 * MB, 4 * GB + 1, 64 * MB))
        + list(range(4 * GB, max(max_size, 4 * GB) + 1, 512 * MB))
    )
    return [s for s in sorted(set(space)) if s <= max_size]


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def results_to_dataframe(all_results: List[dict]):
    """Flatten time_stats into the time_stats.<op>.<stat> columns the predictor reads."""
    import pandas as pd

    all_results = [r for r in all_results if r]
    if not all_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_results)
    return (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
