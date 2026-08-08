"""
merge.py  —  build the merged, arrival-ordered trace

Concatenates session_traces/*.csv into one CSV, shuffled with --seed, and fills
inter_request_latency from the tool benchmark in raw/tool/.

  --mode set   one run per question, from selected_set.txt
  --replicas   repeat the sequence N times (e.g. 3x); analyse the middle one
"""

import argparse
import ast
import csv
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE        = Path(__file__).parent          # GAIATrace/owl/traces
DATA        = BASE.parent                    # GAIATrace/owl
TRACES_DIR  = BASE / "session_traces"
BENCH_DIR   = DATA / "raw" / "tool"          # measured tool latencies (merge input)
MERGE_OUT   = BASE                           # owl_random.csv lands beside the traces
MANIFEST    = BASE / "selected_set.txt"
MERGE_OUT.mkdir(parents=True, exist_ok=True)

# ── Token / model helpers ──────────────────────────────────────────────────────

def model_id_from_token(token_ids_str: str) -> int:
    if token_ids_str.startswith("[200006, 77944"):
        return 1   # gpt-4o
    if token_ids_str.startswith("[200006, 17360"):
        return 0   # gpt-oss-120b
    return -1


def load_tool_latency() -> dict:
    """
    Load all raw/tool/*.csv files written by agent/tool_bench.py.
    Returns {(source_file, request_idx): effective_elapsed}: each tool's latency
    averaged over its benchmark reps, then summed across the tools issued by one
    LLM response. The agent awaits each tool call inside its request loop, so a
    turn's tool calls are serialized and the request waits for their total.
    """
    per_tool: dict[tuple, list] = defaultdict(list)
    for bench_path in sorted(BENCH_DIR.glob('*.csv')):
        for row in csv.DictReader(open(bench_path, encoding='utf-8')):
            try:
                eff = float(row['effective_elapsed'])
                if eff <= 0:
                    continue
                key = (row['source_file'], int(row['request_idx']), row.get('args_json', ''))
                per_tool[key].append(eff)
            except (ValueError, KeyError):
                pass
    # Collapse: mean per tool, then sum across the tools of one request
    result: dict[tuple, float] = defaultdict(float)
    for (src, idx, _args), vals in per_tool.items():
        result[(src, idx)] += sum(vals) / len(vals)
    return dict(result)


MODE_CHOICES = ["full", "set"]


def parse_replicas(value: str) -> int:
    """Accept '3x', '3X', or plain '3'; return the integer."""
    s = value.rstrip("xX")
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError("--replicas must be >= 1")
    return n


# ── Main ───────────────────────────────────────────────────────────────────────

def merge_csv_files(mode: str, output_file: str, seed: int, replicas: int = 1):
    tool_latency = load_tool_latency()
    print(f"Loaded {len(tool_latency)} tool latency entries from raw/tool")

    # Collect all stems
    all_csv = sorted(f for f in TRACES_DIR.glob("*.csv") if not f.stem.endswith("_tools"))
    all_stems = [f.stem for f in all_csv]
    print(f"Found {len(all_stems)} CSV files in {TRACES_DIR}")

    rng = random.Random(seed)

    # Mode selection
    if mode == "set":
        manifest = MANIFEST
        if not manifest.exists():
            raise SystemExit("selected_set.txt not found — run select_set.py first")
        set_stems = set(manifest.read_text(encoding="utf-8").split())
        stems = [s for s in all_stems if s in set_stems]
        print(f"  Set mode: {len(stems)} stems from selected_set.txt")
    else:
        stems = all_stems

    rng.shuffle(stems)
    print(f"Shuffled {len(stems)} files (seed={seed})")

    # Replication: repeat the ordered sequence N times
    if replicas > 1:
        print(f"Replicating x{replicas}  ({len(stems)} → {len(stems) * replicas} files)")
        stems = stems * replicas

    header = [
        "arrived_at",
        "num_prefill_tokens",
        "num_decode_tokens",
        "block_size",
        "request_id",
        "model_id",
        "session_id",
        "turn_id",
        "source_file",
        "dep",
        "inter_request_latency",
        "token_ids",
    ]

    out_path = Path(output_file)
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        session_id = 0
        for file_index, stem in enumerate(stems):
            csv_path  = TRACES_DIR / f"{stem}.csv"
            file_name = csv_path.name
            print(f"  [{file_index:3d}] {file_name}")

            with open(csv_path, "r", encoding="utf-8") as in_f:
                reader  = csv.DictReader(in_f)
                turn_id = 0   # position of this request within the session
                for row in reader:
                    num_prefill   = row["num_prefill_tokens"]
                    num_decode    = row["num_decode_tokens"]
                    token_ids_str = row["tokens"]

                    if int(num_decode) == 0 and int(num_prefill) == 0:
                        continue

                    current_model_id = model_id_from_token(token_ids_str)
                    request_id = f"{file_index}{turn_id:03d}"

                    dep_str = row.get("dep", "[]")
                    inter_latency = 0.0
                    try:
                        deps = ast.literal_eval(dep_str)
                        for dep_idx in deps:
                            val = tool_latency.get((stem, dep_idx))
                            if val is not None:
                                inter_latency = max(inter_latency, val)
                    except Exception:
                        pass

                    writer.writerow([
                        0.0,
                        num_prefill,
                        num_decode,
                        16,
                        request_id,
                        current_model_id,
                        session_id,
                        turn_id,
                        file_name,
                        dep_str,
                        inter_latency,
                        token_ids_str,
                    ])
                    turn_id += 1

            session_id += 1

    print(f"\nDone. {len(stems)} files → {out_path}  (mode={mode}, replicas={replicas}x)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge session_traces CSVs into one trace.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=MODE_CHOICES,
        default="set",
        help=(
            "full  Include all CSV files (multiple attempts per question)\n"
            "set   Select one CSV per question"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=str(MERGE_OUT / f"owl_random.csv"),
        help="Output file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--replicas",
        type=parse_replicas,
        default=1,
        metavar="Nx",
        help=(
            "Repeat the ordered trace sequence N times (e.g. 1x, 3x, 5x).\n"
            "Use the middle replica for analysis to avoid cold-start / tail effects.\n"
        ),
    )
    args = parser.parse_args()
    merge_csv_files(args.mode, args.output, args.seed, args.replicas)
