"""
table1_miro.py — MiroThinker counterpart of owl/analysis/table1_owl.py

Token statistics over all session traces (103 tasks; MiroThinker has no
selection stage — every task is one run).

Two row kinds per task:
  main        fine-tuned Qwen-32B; emits <think>…</think> reasoning tokens
  summarizer  gpt-4o-mini; condenses tool output, no reasoning

Both intra-session (cache reset per task) and inter-session (cache shared across
all tasks) hit rates are reported, matching table1_owl.py.

Outputs: outputs/table1_miro.txt
"""

import argparse
import csv
import math
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument("--block-size", type=int, default=1,
                    help="KV cache block size (1 = token-level, 16 = simulator default)")
args = parser.parse_args()
BLOCK_SIZE = args.block_size

BASE       = Path(__file__).resolve().parent # mirothinker/analysis
DATA       = BASE.parent                     # mirothinker
TRACES_DIR = DATA / "traces" / "session_traces"
OUT_DIR    = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KINDS = ("main", "summarizer")
KIND_NAME = {"main": "main  (Qwen-32B fine-tuned)",
             "summarizer": "summarizer  (gpt-4o-mini)"}

THINK_OPEN  = 151667   # <think>
THINK_CLOSE = 151668   # </think>


# ── Helpers (block-hash cache: identical to table1_owl.py) ─────────────────

def parse_token_ids(tok_str):
    return list(map(int, tok_str[1:-1].split(", ")))


def session_cache_hits(token_ids, prefill, block_cache, block_size):
    """
    Cached prefill tokens via a block-hash prefix cache.
    Stops at the first cache miss (prefix-cache semantics).
    """
    num_cached = 0
    parent_hash = 0
    for start in range(0, prefill, block_size):
        end = start + block_size
        if end > prefill:
            break  # partial final block is not cacheable; also keeps cached <= prefill
        block = tuple(token_ids[start:end])
        key = hash((parent_hash, block))
        if key in block_cache:
            num_cached += block_size
            parent_hash = key
        else:
            break
    return num_cached


def session_cache_add(token_ids, block_cache, block_size):
    """Add all complete blocks from token_ids (prefill+decode) into the cache."""
    parent_hash = 0
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        block = tuple(token_ids[start:start + block_size])
        key = hash((parent_hash, block))
        block_cache.add(key)
        parent_hash = key


def reasoning_count(token_ids, prefill, decode):
    """Tokens from decode start up to and including the first </think>."""
    dec = token_ids[prefill: prefill + decode]
    if not dec or dec[0] != THINK_OPEN:
        return 0
    try:
        return dec.index(THINK_CLOSE) + 1
    except ValueError:
        return decode  # whole decode is reasoning (no close tag found)


def _stats(vals):
    if not vals:
        return None
    mn, mx = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    return mn, mean, std, mx


# ── Collect ────────────────────────────────────────────────────────────────

csv_files = sorted(TRACES_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit(f"no traces in {TRACES_DIR}")
print(f"Loaded {len(csv_files)} session traces from {TRACES_DIR}")

skipped = 0   # rows excluded by the filters below; expected 0 on the shipped data

acc = {k: {"turns": [], "prefills": [], "decodes": [],
           "cached": [], "unseen": [], "cached_global": [], "reasoning": []}
       for k in KINDS}

# Never reset: lets a later session hit blocks cached by an earlier one, so the
# difference from the per-session figure is the cross-session contribution.
global_cache = {k: set() for k in KINDS}

for path in csv_files:
    file_rows = {k: 0 for k in KINDS}
    # Per-kind block cache, reset each task (infinite intra-session cache).
    block_cache = {k: set() for k in KINDS}

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kind = row.get("row_kind", "")
            if kind not in KINDS:
                skipped += 1
                continue
            try:
                p = int(row["num_prefill_tokens"])
                d = int(row["num_decode_tokens"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            tok_str = row.get("tokens", "")
            if p < 0 or (p == 0 and d == 0) or not tok_str.startswith("["):
                skipped += 1
                continue

            token_ids = parse_token_ids(tok_str)
            cached = session_cache_hits(token_ids, p, block_cache[kind], BLOCK_SIZE)
            cached_g = session_cache_hits(token_ids, p, global_cache[kind], BLOCK_SIZE)
            session_cache_add(token_ids, block_cache[kind], BLOCK_SIZE)
            session_cache_add(token_ids, global_cache[kind], BLOCK_SIZE)

            acc[kind]["prefills"].append(p)
            acc[kind]["decodes"].append(d)
            acc[kind]["cached"].append(cached)
            acc[kind]["unseen"].append(p - cached)
            acc[kind]["cached_global"].append(cached_g)
            file_rows[kind] += 1

            if kind == "main":
                acc["main"]["reasoning"].append(reasoning_count(token_ids, p, d))

    for k in KINDS:
        if file_rows[k] > 0:
            acc[k]["turns"].append(file_rows[k])


# ── Format & write text report ────────────────────────────────────────────

lines = []
lines.append("=" * 64)
lines.append("  MIROTHINKER SESSION STATISTICS")
lines.append(f"  {len(csv_files)} tasks (no selection stage — one run each)")
lines.append(f"  block-size = {BLOCK_SIZE}   (sessions walked in sorted filename order)")
if skipped:
    lines.append(f"  WARNING: {skipped} rows excluded (unknown row_kind or unusable token counts)")
lines.append("=" * 64)

for kind in KINDS:
    a = acc[kind]
    n = len(a["prefills"])
    if n == 0:
        continue

    t = _stats(a["turns"])
    p = _stats(a["prefills"])
    d = _stats(a["decodes"])

    total_p  = sum(a["prefills"])
    total_c  = sum(a["cached"])
    total_u  = sum(a["unseen"])
    hit_rate = total_c / total_p if total_p else 0.0

    lines.append("")
    lines.append(f"  {KIND_NAME[kind]}")
    lines.append(f"  {'─'*56}")
    lines.append(f"  Requests      : {n:,}  across {len(a['turns'])} files")
    lines.append(f"  Turns/file    : min={t[0]}  mean={t[1]:.1f}±{t[2]:.1f}  max={t[3]}")
    lines.append(f"  Prefill (tok) : min={p[0]:,}  mean={p[1]:,.1f}±{p[2]:,.1f}  max={p[3]:,}")
    lines.append(f"  Decode  (tok) : min={d[0]:,}  mean={d[1]:,.1f}±{d[2]:,.1f}  max={d[3]:,}")
    total_cg = sum(a["cached_global"])
    hit_g    = total_cg / total_p if total_p else 0.0

    lines.append(f"  KV cache (block={BLOCK_SIZE})")
    lines.append(f"    total prefill : {total_p:>14,}")
    lines.append(f"    intra-session : {total_c:>14,}  ({hit_rate*100:.1f}% hit rate)  cache reset per session")
    lines.append(f"    + inter       : {total_cg:>14,}  ({hit_g*100:.1f}% hit rate)  cache shared across all sessions")
    lines.append(f"    cross-session : {total_cg-total_c:>14,}  (+{(hit_g-hit_rate)*100:.1f} pp from reuse between sessions)")
    lines.append(f"    unseen (intra): {total_u:>14,}  ({(1-hit_rate)*100:.1f}%)")

    if kind == "main" and a["reasoning"]:
        r = _stats(a["reasoning"])
        r_frac = sum(a["reasoning"]) / sum(a["decodes"])
        lines.append(f"  Reasoning (tok): min={r[0]:,}  mean={r[1]:,.1f}±{r[2]:,.1f}  max={r[3]:,}  "
                     f"({r_frac*100:.1f}% of decode)")
    elif kind == "summarizer":
        lines.append(f"  Reasoning (tok): N/A (gpt-4o-mini has no reasoning tokens)")

lines.append("")
lines.append("=" * 64)

report = "\n".join(lines)
print(report)

out = OUT_DIR / "table1_miro.txt"
out.write_text(report + "\n", encoding="utf-8")
print(f"\nWritten: {out}")
