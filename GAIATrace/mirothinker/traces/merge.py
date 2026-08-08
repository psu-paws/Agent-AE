"""
merge.py

Merges the per-task CSVs in session_traces/ into one shuffled CSV for the Vidur
simulator, and fills inter_request_latency from the tool benchmark in raw/tool/.

One CSV per GAIA task. Rows with row_kind='summarizer_placeholder' or
num_prefill_tokens=-1 are skipped.
"""

import argparse
import csv
import re
import random
import sys
from collections import defaultdict
from pathlib import Path

# Only include per-task CSVs (UUID-named); excludes merged outputs, summarizer files, etc.
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
)

csv.field_size_limit(sys.maxsize)

BASE       = Path(__file__).parent           # GAIATrace/mirothinker/traces
DATA       = BASE.parent                     # GAIATrace/mirothinker
TRACES_DIR = BASE / "session_traces"         # per-task CSVs (main + summarizer rows)
BENCH_DIR  = DATA / "raw" / "tool"           # measured MCP tool latencies
MERGE_OUT  = BASE                            # miro_random.csv lands beside the traces


def load_tool_latency() -> dict:
    """
    Reads raw/tool/*.csv written by agent/apps/miroflow-agent/benchmarks/.
    Returns {(uuid_tools_filename, turn): latency} where latency is
    max over parallel tools at the same turn, each averaged over iterations.
    """
    per_tool: dict[tuple, list] = defaultdict(list)
    if not BENCH_DIR.exists():
        return {}
    for bench_path in sorted(BENCH_DIR.glob("*.csv")):
        for row in csv.DictReader(open(bench_path, encoding="utf-8")):
            try:
                runtime = float(row["runtime_s"])
                if runtime <= 0:
                    continue
                key = (row["filename"], int(row["turn"]), row.get("args_json", ""))
                per_tool[key].append(runtime)
            except (ValueError, KeyError):
                pass
    result: dict[tuple, float] = defaultdict(float)
    for (fname, turn, _args), vals in per_tool.items():
        tool_mean = sum(vals) / len(vals)
        result[(fname, turn)] = max(result[(fname, turn)], tool_mean)
    print(f"Loaded {len(result)} tool latency entries from {BENCH_DIR}")
    return dict(result)


SKIP_KINDS = {"summarizer_placeholder"}


# ── Model detection ────────────────────────────────────────────────────────────

def model_id_from_token(token_ids_str: str) -> int:
    if token_ids_str.startswith("[151644, 8948"):
        return 1   # main (Qwen)
    if token_ids_str.startswith("[200006, 1428"):
        return 0   # summarizer (gpt-4o-mini)
    return -1


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_csv_files(input_dir: Path, output_file: Path, seed: int):
    tool_latency = load_tool_latency()

    all_csv   = sorted(input_dir.glob("*.csv"))
    all_stems = [f.stem for f in all_csv if _UUID_RE.match(f.stem)]
    print(f"Found {len(all_stems)} CSV files in {input_dir}")

    rng   = random.Random(seed)
    stems = list(all_stems)
    rng.shuffle(stems)

    header = [
        "arrived_at",
        "num_prefill_tokens",
        "num_decode_tokens",
        "block_size",
        "request_id",
        "model_id",
        "session_id",
        "source_file",
        "inter_request_latency",
        "token_ids",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        request_id = 0
        for file_index, stem in enumerate(stems):
            csv_path = input_dir / f"{stem}.csv"
            print(f"  [{file_index:3d}] {csv_path.name}")

            tools_filename = f"{stem}_tools.csv"
            with open(csv_path, encoding="utf-8") as in_f:
                row_index  = 0     # position among emitted rows (used for request_id)
                main_index = 0     # agent turn number: only main rows are turns
                pending    = None  # tool turn issued but not yet consumed
                for row in csv.DictReader(in_f):
                    if row.get("row_kind") in SKIP_KINDS:
                        continue
                    try:
                        p = int(row["num_prefill_tokens"])
                        d = int(row["num_decode_tokens"])
                    except (ValueError, KeyError):
                        continue
                    if p < 0 or d <= 0:
                        continue
                    token_ids_str = row.get("tokens", "")
                    model_id = model_id_from_token(token_ids_str)
                    if model_id == -1:
                        continue
                    inter_latency = (tool_latency.get((tools_filename, pending), 0.0)
                                     if pending is not None else 0.0)
                    pending = None
                    if row.get("row_kind") == "main":
                        pending = main_index
                        main_index += 1

                    writer.writerow([
                        0.0,
                        p,
                        d,
                        16,
                        f"{file_index}{row_index:03d}",
                        model_id,
                        request_id,
                        csv_path.name,
                        inter_latency,
                        token_ids_str,
                    ])
                    row_index += 1

            request_id += 1

    print(f"\nDone. {len(stems)} files → {output_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge per-task CSVs into a single Vidur-compatible CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_dir", nargs="?", default=str(TRACES_DIR),
        help="Directory containing the per-task CSVs (default: current dir)",
    )
    parser.add_argument(
        "--output", "-o", default=str(MERGE_OUT / "miro_random.csv"),
        help="Output file path (default: miro_random.csv beside the traces)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_path  = Path(args.output)

    merge_csv_files(input_dir, out_path, args.seed)
