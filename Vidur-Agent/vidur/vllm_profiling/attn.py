"""Attention profiler, emitting `attention.csv`.

Times FlashInfer's paged prefill and decode wrappers and vLLM's
reshape_and_cache_flash, giving the columns attn_prefill, attn_decode and
attn_kv_cache_save.

    python -m vidur.vllm_profiling.attn --model <hf-model> \
        --num_tensor_parallel_workers 1 2 --max_seq_len 16384
"""

import argparse
import datetime
import os
from dataclasses import dataclass
from itertools import product
from math import ceil, floor
from typing import List

import torch
from tqdm import tqdm

from .common import (
    ACTIVE_STEPS,
    WARMUP_STEPS,
    CudaTimer,
    ModelConfig,
    TimerStatsStore,
    get_attention_batch_sizes_to_profile,
    get_attention_prefill_chunk_sizes_to_profile,
    get_seq_lengths_to_profile,
    results_to_dataframe,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class AttentionInput:
    prefill_chunk_size: int
    kv_cache_size: int
    batch_size: int
    is_prefill: bool

    def is_valid(self, max_seq_len: int) -> bool:
        total = self.prefill_chunk_size + self.kv_cache_size
        if total > max_seq_len:
            return False
        if self.is_prefill:
            # vidur only profiles single-sequence prefills
            return self.batch_size == 1 and self.prefill_chunk_size > 0
        return self.prefill_chunk_size == 0 and self.kv_cache_size > 0

    def is_under_memory_limit(self, max_tokens: int) -> bool:
        return (
            self.batch_size * (self.kv_cache_size + self.prefill_chunk_size)
            <= max_tokens
        )


def get_attention_input_combinations(
    max_seq_len, min_bs, max_bs, only_prefill, only_decode, max_chunk
) -> List[AttentionInput]:
    combos = []
    for chunk in get_attention_prefill_chunk_sizes_to_profile(max_seq_len, max_chunk):
        num_partitions = max_seq_len // chunk
        kv_sizes = [i * chunk for i in range(num_partitions)]
        combos.extend(product([chunk], kv_sizes, [1], [True]))
    if max_seq_len <= max_chunk:
        combos.extend(product(get_seq_lengths_to_profile(max_seq_len), [0], [1], [True]))
    combos.extend(
        product(
            [0],
            get_seq_lengths_to_profile(max_seq_len),
            get_attention_batch_sizes_to_profile(min_bs, max_bs),
            [False],
        )
    )

    out = []
    for chunk, kv, bs, is_prefill in combos:
        if is_prefill and only_decode:
            continue
        if not is_prefill and only_prefill:
            continue
        ai = AttentionInput(chunk, kv, bs, is_prefill)
        if ai.is_valid(max_seq_len):
            out.append(ai)
    return out


def get_max_num_blocks(cfg: ModelConfig, tp: int, block_size: int, dtype, util=0.9) -> int:
    elem = torch.randn(1, dtype=dtype).element_size()
    per_block = (
        2 * block_size * cfg.num_kv_heads_per_worker(tp) * cfg.head_size * elem
    )
    total = per_block * cfg.num_layers
    return floor((torch.cuda.mem_get_info()[1] * util) / total)


class FlashInferAttentionProfiler:
    def __init__(self, cfg: ModelConfig, tp: int, max_num_blocks: int,
                 max_model_len: int, block_size: int, dtype, kv_layout: str = "HND"):
        import flashinfer

        self.cfg, self.tp, self.dtype = cfg, tp, dtype
        self.block_size, self.max_model_len = block_size, max_model_len
        self.kv_layout = kv_layout
        self.store = TimerStatsStore()
        self.device = torch.device("cuda")

        self.n_q = cfg.num_q_heads_per_worker(tp)
        self.n_kv = cfg.num_kv_heads_per_worker(tp)
        self.hd = cfg.head_size
        self.max_num_blocks = max_num_blocks

        # NHD: [pages, 2, page_size, kv_heads, head_dim]
        # HND: [pages, 2, kv_heads, page_size, head_dim]
        kv_shape = ((max_num_blocks, 2, block_size, self.n_kv, self.hd)
                    if kv_layout == "NHD" else
                    (max_num_blocks, 2, self.n_kv, block_size, self.hd))
        self.kv_cache = torch.randn(*kv_shape, dtype=dtype, device=self.device)
        # vLLM's reshape_and_cache_flash writes into separate k/v caches
        self.k_cache = torch.zeros(
            max_num_blocks, block_size, self.n_kv, self.hd, dtype=dtype, device=self.device
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # Allocated once: creating these inside the timed region measures a
        # host-to-device allocation rather than the kernel.
        self.k_scale = torch.tensor(1.0, device=self.device)
        self.v_scale = torch.tensor(1.0, device=self.device)
        from vllm import _custom_ops as _ops
        self._ops = _ops

        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout)
        # Matches vLLM's decode wrapper, which enables tensor cores by default.
        # Also required for GQA group sizes the cuda-cores template does not
        # dispatch, since it selects group_size at compile time.
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout, use_tensor_cores=True
        )

    def _t(self, name):
        return CudaTimer(name, self.store)

    def _plan_and_inputs(self, ai: AttentionInput):
        bs = ai.batch_size
        q_len = ai.prefill_chunk_size if ai.is_prefill else 1
        total_kv = ai.kv_cache_size + q_len
        pages_per_seq = ceil(total_kv / self.block_size)

        if pages_per_seq * bs > self.max_num_blocks:
            return None

        qo_indptr = torch.arange(0, (bs + 1) * q_len, q_len, dtype=torch.int32, device=self.device)
        kv_indptr = torch.arange(0, (bs + 1) * pages_per_seq, pages_per_seq,
                                 dtype=torch.int32, device=self.device)
        kv_indices = torch.arange(bs * pages_per_seq, dtype=torch.int32, device=self.device)
        last_page = total_kv - (pages_per_seq - 1) * self.block_size
        kv_last_page_len = torch.full((bs,), last_page, dtype=torch.int32, device=self.device)

        q = torch.randn(bs * q_len, self.n_q, self.hd, dtype=self.dtype, device=self.device)
        k = torch.randn(bs * q_len, self.n_kv, self.hd, dtype=self.dtype, device=self.device)
        v = torch.randn_like(k)
        slot_mapping = torch.arange(bs * q_len, dtype=torch.int64, device=self.device)

        if ai.is_prefill:
            self.prefill_wrapper.plan(
                qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                self.n_q, self.n_kv, self.hd, self.block_size,
                causal=True, q_data_type=self.dtype, kv_data_type=self.dtype,
            )
        else:
            self.decode_wrapper.plan(
                kv_indptr, kv_indices, kv_last_page_len,
                self.n_q, self.n_kv, self.hd, self.block_size,
                q_data_type=self.dtype, kv_data_type=self.dtype,
            )
        return q, k, v, slot_mapping

    def _step(self, ai: AttentionInput, q, k, v, slot_mapping):
        with self._t("attn_kv_cache_save"):
            self._ops.reshape_and_cache_flash(
                k, v, self.k_cache, self.v_cache, slot_mapping, "auto",
                self.k_scale, self.v_scale,
            )
        name = "attn_prefill" if ai.is_prefill else "attn_decode"
        with self._t(name):
            if ai.is_prefill:
                self.prefill_wrapper.run(q, self.kv_cache)
            else:
                self.decode_wrapper.run(q, self.kv_cache)

    @torch.inference_mode()
    def profile(self, ai: AttentionInput):
        prepared = self._plan_and_inputs(ai)
        if prepared is None:
            return None
        q, k, v, slot_mapping = prepared

        for _ in range(WARMUP_STEPS):
            self._step(ai, q, k, v, slot_mapping)
        torch.cuda.synchronize()

        self.store.clear_stats()
        for _ in range(ACTIVE_STEPS):
            self._step(ai, q, k, v, slot_mapping)
        torch.cuda.synchronize()
        stats = self.store.get_stats()
        self.store.clear_stats()

        return {
            "time_stats": stats,
            "n_embd": self.cfg.embedding_dim,
            "n_q_head": self.cfg.num_q_heads,
            "n_kv_head": self.cfg.num_kv_heads,
            "block_size": self.block_size,
            "num_tensor_parallel_workers": self.tp,
            "max_model_len": self.max_model_len,
            "batch_size": ai.batch_size,
            "prefill_chunk_size": ai.prefill_chunk_size,
            "kv_cache_size": ai.kv_cache_size,
            "is_prefill": ai.is_prefill,
            "attention_backend": "FLASHINFER",
            "kv_layout": self.kv_layout,
        }


def parse_args():
    p = argparse.ArgumentParser(description="vLLM/FlashInfer attention profiling")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--num_tensor_parallel_workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--max_seq_len", type=int, default=131072)
    p.add_argument("--min_batch_size", type=int, default=1)
    p.add_argument("--max_batch_size", type=int, default=128,
                   help="Should match the serving max_num_seqs, which caps the decode batch.")
    p.add_argument("--max_chunk_size", type=int, default=8192)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--kv-layout", choices=["NHD","HND"], default="HND",
                   help="KV cache layout, which changes the memory access pattern. "
                        "Match the layout vLLM logs at startup.")
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    p.add_argument("--profile_only_prefill", action="store_true")
    p.add_argument("--profile_only_decode", action="store_true")
    p.add_argument("--output_dir", default="profiling_outputs")
    a = p.parse_args()
    a.output_dir = f"{a.output_dir}/attention/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(a.output_dir, exist_ok=True)
    return a


def main():
    args = parse_args()
    dtype = DTYPES[args.dtype]
    cfg = ModelConfig.from_model_name(args.model)

    combos = get_attention_input_combinations(
        args.max_seq_len, args.min_batch_size, args.max_batch_size,
        args.profile_only_prefill, args.profile_only_decode, args.max_chunk_size,
    )
    print(f"{args.model}: {len(combos)} input combinations")

    all_results = []
    for tp in args.num_tensor_parallel_workers:
        max_blocks = get_max_num_blocks(cfg, tp, args.block_size, dtype)
        # cap so the KV tensors themselves do not eat the whole card
        max_blocks = min(max_blocks, 200_000)
        prof = FlashInferAttentionProfiler(
            cfg, tp, max_blocks, args.max_seq_len, args.block_size, dtype, args.kv_layout
        )
        usable = [c for c in combos if c.is_under_memory_limit(max_blocks * args.block_size)]
        print(f"  tp={tp}: max_num_blocks={max_blocks}, {len(usable)} usable combos")
        for ai in tqdm(usable, desc=f"attn tp={tp}"):
            try:
                r = prof.profile(ai)
                if r:
                    all_results.append(r)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    skip {ai}: {type(e).__name__}: {str(e)[:100]}")
        del prof
        torch.cuda.empty_cache()

    df = results_to_dataframe(all_results)
    out_dir = f"{args.output_dir}/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/attention.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
