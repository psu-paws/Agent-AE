"""Linear-layer profiler, emitting `mlp.csv`.

Times the ops vLLM dispatches, built inside a VllmConfig context so the custom
CUDA kernels are used rather than the torch fallbacks. Columns, keyed on
num_tokens: attn_pre_proj, attn_post_proj, attn_rope, add, mlp_up_proj,
mlp_down_proj, mlp_act, input_layernorm, post_attention_layernorm.

Tensor parallelism is emulated on one GPU by sharding head counts and the MLP
hidden dim; all-reduce cost is profiled separately by collective.py.

    python -m vidur.vllm_profiling.mlp --model <hf-model> \
        --num_tensor_parallel_workers 1 2 --max_tokens 8192
"""

import argparse
import datetime
import os
from typing import List

import torch
from tqdm import tqdm

from .common import (
    ACTIVE_STEPS,
    WARMUP_STEPS,
    CudaTimer,
    ModelConfig,
    TimerStatsStore,
    get_num_tokens_to_profile,
    results_to_dataframe,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


class ProfiledBlock(torch.nn.Module):
    """One transformer block's non-attention work, with each op individually timed.

    The op names match vidur's exactly so the CSV columns line up.
    """

    def __init__(self, cfg: ModelConfig, tp: int, store: TimerStatsStore, dtype,
                 rope_yarn_factor=None):
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding import get_rope

        self.cfg, self.tp, self.store = cfg, tp, store

        n_q = cfg.num_q_heads_per_worker(tp)
        n_kv = cfg.num_kv_heads_per_worker(tp)
        hd = cfg.head_size
        self.q_size, self.kv_size = n_q * hd, n_kv * hd
        mlp_hidden = cfg.mlp_hidden_per_worker(tp)

        dev = torch.device("cuda")
        lin = lambda i, o, b: torch.nn.Linear(i, o, bias=b, device=dev, dtype=dtype)

        # attn_pre_proj == fused QKV projection (vLLM uses QKVParallelLinear)
        self.qkv_proj = lin(cfg.embedding_dim, self.q_size + 2 * self.kv_size, cfg.use_qkv_bias)
        # attn_post_proj == output projection (RowParallelLinear)
        self.o_proj = lin(n_q * hd, cfg.embedding_dim, cfg.use_bias)
        # gated MLP: up_proj emits 2x hidden (gate+up), matching MergedColumnParallelLinear
        self.up_proj = lin(cfg.embedding_dim, (2 if cfg.use_gated_mlp else 1) * mlp_hidden, cfg.use_bias)
        self.down_proj = lin(mlp_hidden, cfg.embedding_dim, cfg.use_bias)

        self.act = SiluAndMul() if cfg.use_gated_mlp else torch.nn.GELU()
        self.input_layernorm = RMSNorm(cfg.embedding_dim, eps=cfg.rms_norm_eps).to(dev, dtype)
        self.post_attention_layernorm = RMSNorm(cfg.embedding_dim, eps=cfg.rms_norm_eps).to(dev, dtype)

        rope_params = {"rope_theta": cfg.rope_theta, "rope_type": "default"}
        max_pos = cfg.max_position_embeddings
        if rope_yarn_factor:
            # rope_parameters/rope_type spelling, not rope_scaling/type.
            rope_params = {"rope_theta": cfg.rope_theta, "rope_type": "yarn",
                           "factor": rope_yarn_factor,
                           "original_max_position_embeddings": cfg.max_position_embeddings}
            max_pos = int(cfg.max_position_embeddings * rope_yarn_factor)
        self.rotary_emb = get_rope(
            head_size=hd,
            max_position=max_pos,
            is_neox_style=True,
            rope_parameters=rope_params,
            dtype=dtype,
        )

    def _t(self, name: str) -> CudaTimer:
        return CudaTimer(name, self.store)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        residual = hidden_states
        with self._t("input_layernorm"):
            hidden_states = self.input_layernorm(hidden_states)

        with self._t("attn_pre_proj"):
            qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        with self._t("attn_rope"):
            q, k = self.rotary_emb(positions, q, k)

        # attention itself is measured by attn.py; stand in with a same-shaped tensor
        attn_output = torch.empty_like(q)
        with self._t("attn_post_proj"):
            hidden_states = self.o_proj(attn_output)

        hidden_states = residual + hidden_states
        residual = hidden_states

        with self._t("post_attention_layernorm"):
            hidden_states = self.post_attention_layernorm(hidden_states)

        with self._t("mlp_up_proj"):
            hidden_states = self.up_proj(hidden_states)
        with self._t("mlp_act"):
            hidden_states = self.act(hidden_states)
        with self._t("mlp_down_proj"):
            hidden_states = self.down_proj(hidden_states)

        with self._t("add"):
            hidden_states = residual + hidden_states
        return hidden_states


def profile_one(block: ProfiledBlock, cfg: ModelConfig, tp: int, num_tokens: int, dtype) -> dict:
    store = block.store
    hidden = torch.randn(num_tokens, cfg.embedding_dim, device="cuda", dtype=dtype)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)

    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            block(hidden, positions)
        torch.cuda.synchronize()

        store.clear_stats()
        for _ in range(ACTIVE_STEPS):
            block(hidden, positions)
        torch.cuda.synchronize()
        time_stats = store.get_stats()
    store.clear_stats()

    return {
        "time_stats": time_stats,
        "n_head": cfg.num_q_heads,
        "n_kv_head": cfg.num_kv_heads,
        "n_embd": cfg.embedding_dim,
        "n_expanded_embd": cfg.mlp_hidden_dim,
        "vocab_size": cfg.vocab_size,
        "use_gated_mlp": cfg.use_gated_mlp,
        "num_tokens": num_tokens,
        "num_tensor_parallel_workers": tp,
    }


def parse_args():
    p = argparse.ArgumentParser(description="vLLM MLP profiling (vidur-compatible output)")
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--num_tensor_parallel_workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--max_tokens", type=int, default=65536)
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    p.add_argument("--rope-yarn-factor", type=float, default=None,
                   help="Enable static YaRN with this factor. Must match the serving "
                        "config: YaRN uses a different RotaryEmbedding class.")
    p.add_argument("--output_dir", default="profiling_outputs")
    p.add_argument(
        "--eager",
        action="store_true",
        default=True,
        help="Match the serving config (--enforce-eager): no torch.compile/CUDA graphs.",
    )
    a = p.parse_args()
    a.output_dir = f"{a.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(a.output_dir, exist_ok=True)
    return a


def main():
    args = parse_args()
    dtype = DTYPES[args.dtype]
    torch.set_default_dtype(dtype)

    cfg = ModelConfig.from_model_name(args.model)
    print(f"{args.model}: embd={cfg.embedding_dim} mlp_hidden={cfg.mlp_hidden_dim} "
          f"q_heads={cfg.num_q_heads} kv_heads={cfg.num_kv_heads} layers={cfg.num_layers}")

    # vLLM layers resolve their custom-op dispatch through the current config, so
    # build a real one for this model and keep it entered while constructing.
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs

    # Fusion is disabled so each op can be timed separately: with norm/act
    # fusion on, RMSNorm and SiluAndMul fold into neighbouring ops and cannot be
    # attributed to the input_layernorm and mlp_act columns.
    from vllm.config.compilation import CompilationConfig, PassConfig

    vllm_cfg = EngineArgs(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=args.eager,
        tensor_parallel_size=1,   # TP is emulated by sharding dims, see module docstring
        load_format="dummy",
        compilation_config=CompilationConfig(
            pass_config=PassConfig(
                fuse_norm_quant=False,
                fuse_act_quant=False,
                fuse_attn_quant=False,
                fuse_rope_kvcache=False,
            )
        ),
    ).create_engine_config()

    num_tokens_list = get_num_tokens_to_profile(args.max_tokens)
    all_results: List[dict] = []

    with set_current_vllm_config(vllm_cfg):
        for tp in args.num_tensor_parallel_workers:
            if cfg.num_q_heads % tp or cfg.mlp_hidden_dim % tp:
                print(f"skip tp={tp}: dims not divisible")
                continue
            store = TimerStatsStore()
            block = ProfiledBlock(cfg, tp, store, dtype, args.rope_yarn_factor)
            for num_tokens in tqdm(num_tokens_list, desc=f"mlp tp={tp}"):
                try:
                    all_results.append(profile_one(block, cfg, tp, num_tokens, dtype))
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM at num_tokens={num_tokens}, tp={tp}; skipping")
                    torch.cuda.empty_cache()
            del block
            torch.cuda.empty_cache()

    df = results_to_dataframe(all_results)
    out_dir = f"{args.output_dir}/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/mlp.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
