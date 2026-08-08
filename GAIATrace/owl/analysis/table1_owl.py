"""
table1_owl.py

Computes token statistics on the "set" selection.
Reads the selected stems from selected_set.txt (built by select_set.py).

Intra-session KV cache hit estimation uses a block-hash prefix cache shared
across all agents within a task file, matching the simulator's algorithm.
--block-size 16  matches the simulator default (block-aligned hits)
--block-size 1   gives token-level matching (no alignment rounding)

Inter-session sharing is not estimated (treated as 0 cached).

Outputs: table1_owl.txt  (KV hit rate via block-hash prefix cache, intra- and inter-session)
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

BASE        = Path(__file__).resolve().parent  # GAIATrace/owl/analysis
DATA        = BASE.parent                    # GAIATrace/owl
TRACES_DIR  = DATA / "traces" / "session_traces"
OUTCOMES    = DATA / "traces" / "run_outcomes.csv"   # per-run solved/not verdict
MANIFEST    = DATA / "traces" / "selected_set.txt"
OUT_DIR     = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load manifest ──────────────────────────────────────────────────────────

_OUTCOMES = None

def get_score(stem):
    """True/False if the run was solved, None if it has no recorded verdict.

    Read from traces/run_outcomes.csv rather than the agent logs: the logs are
    plain-text GAIA content and are not shipped with the artifact (see README).
    The verdict is the only thing this ever needed from them.
    """
    global _OUTCOMES
    if _OUTCOMES is None:
        with open(OUTCOMES, encoding="utf-8") as f:
            _OUTCOMES = {r["source_file"]: r["solved"] == "True"
                         for r in csv.DictReader(f)}
    return _OUTCOMES.get(stem)


# ── Helpers ────────────────────────────────────────────────────────────────

def model_id(tok_str):
    if tok_str.startswith("[200006, 77944"):
        return 1   # gpt-4o
    if tok_str.startswith("[200006, 17360"):
        return 0   # gpt-oss-120b
    return -1


def parse_token_ids(tok_str):
    """Parse '[id, id, ...]' string into a list of ints."""
    return list(map(int, tok_str[1:-1].split(", ")))


def session_cache_hits(token_ids, prefill, block_cache, block_size):
    """
    Count cached prefill tokens using a block-hash prefix cache.
    Returns the number of cached tokens (multiple of block_size, or exact if block_size=1).
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


def reasoning_token_count(tok_str, prefill, decode):
    """
    Parse the full token list, slice the decode portion, find the first
    200007 (<|end|>). Returns count up to and including that token (0 if absent).
    """
    ids = list(map(int, tok_str[1:-1].split(", ")))
    decode_slice = ids[prefill: prefill + decode]
    try:
        return decode_slice.index(200007) + 1
    except ValueError:
        return 0


def _stats(vals):
    if not vals:
        return None
    mn, mx = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    return mn, mean, std, mx


# ── Collect data ───────────────────────────────────────────────────────────

manifest = MANIFEST
if not manifest.exists():
    raise SystemExit("selected_set.txt not found — run select_set.py first")
stems = [s for s in manifest.read_text(encoding="utf-8").splitlines() if s]
print(f"Loaded {len(stems)} stems from selected_set.txt")

acc = {
    mid: {
        "turns":     [],
        "prefills":  [],
        "decodes":   [],
        "cached":    [],
        "unseen":    [],
        "cached_global": [],   # same metric against a cache that never resets
        "reasoning": [],
    }
    for mid in (0, 1)
}

global_cache = {0: set(), 1: set()}

skipped = 0        # rows whose tokens match neither model; expected 0
missing_csv = []   # manifest stems with no trace file

for stem in stems:
    csv_path  = TRACES_DIR / f"{stem}.csv"
    if not csv_path.exists():
        missing_csv.append(stem)
        continue
    file_rows = {0: 0, 1: 0}
    # Per-model block cache, reset each task file (infinite intra-session cache).
    # Shared across all agents of the same model within a task.
    block_cache = {0: set(), 1: set()}

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = int(row["num_prefill_tokens"])
                d = int(row["num_decode_tokens"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            tok_str = row.get("tokens", "")
            mid = model_id(tok_str)
            if mid == -1:
                skipped += 1
                continue

            token_ids = parse_token_ids(tok_str)
            cached = session_cache_hits(token_ids, p, block_cache[mid], BLOCK_SIZE)
            cached_g = session_cache_hits(token_ids, p, global_cache[mid], BLOCK_SIZE)
            session_cache_add(token_ids, block_cache[mid], BLOCK_SIZE)
            session_cache_add(token_ids, global_cache[mid], BLOCK_SIZE)

            unseen = p - cached
            acc[mid]["prefills"].append(p)
            acc[mid]["decodes"].append(d)
            acc[mid]["cached"].append(cached)
            acc[mid]["unseen"].append(unseen)
            acc[mid]["cached_global"].append(cached_g)
            file_rows[mid] += 1

            if mid == 0:
                acc[0]["reasoning"].append(reasoning_token_count(tok_str, p, d))

    for mid in (0, 1):
        if file_rows[mid] > 0:
            acc[mid]["turns"].append(file_rows[mid])


# ── Format & write text report ────────────────────────────────────────────

MODEL_NAME = {0: "gpt-oss-120b", 1: "gpt-4o"}

lines = []
lines.append("=" * 64)
lines.append("  SET STATISTICS  (one best run per question, seed=42)")
lines.append(f"  {len(stems)} questions selected")
lines.append(f"  block-size = {BLOCK_SIZE}   (sessions walked in manifest order)")
if missing_csv:
    lines.append(f"  WARNING: {len(missing_csv)} manifest stems have no trace CSV and were dropped")
if skipped:
    lines.append(f"  WARNING: {skipped} rows excluded (tokens match neither model)")
lines.append("=" * 64)

for mid, name in MODEL_NAME.items():
    a = acc[mid]
    n = len(a["prefills"])
    if n == 0:
        continue

    t  = _stats(a["turns"])
    p  = _stats(a["prefills"])
    d  = _stats(a["decodes"])

    total_p  = sum(a["prefills"])
    total_c  = sum(a["cached"])
    total_u  = sum(a["unseen"])
    hit_rate = total_c / total_p if total_p else 0.0

    lines.append("")
    lines.append(f"  {name}")
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

    if mid == 0 and a["reasoning"]:
        r = _stats(a["reasoning"])
        r_frac = sum(a["reasoning"]) / sum(a["decodes"])
        lines.append(f"  Reasoning (tok): min={r[0]:,}  mean={r[1]:,.1f}±{r[2]:,.1f}  max={r[3]:,}  "
                     f"({r_frac*100:.1f}% of decode)")
    elif mid == 1:
        lines.append(f"  Reasoning (tok): N/A (gpt-4o has no reasoning tokens)")

lines.append("")
lines.append("=" * 64)

report = "\n".join(lines)
print(report)

out = OUT_DIR / "table1_owl.txt"
out.write_text(report + "\n", encoding="utf-8")
print(f"\nWritten: {out}")
