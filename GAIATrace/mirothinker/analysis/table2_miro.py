"""
Latency summary for the replayed tool calls

Input:
    mirothinker/raw/tool/{tool_name}.csv   filename, turn, iteration, args_json,
                                           runtime_s, output_20
Usage:
    python table2_miro.py [--iterations 3]
Output:
    outputs/table2_miro.txt
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

BASE     = Path(__file__).resolve().parent      # mirothinker/analysis
DATA     = BASE.parent                          # mirothinker
TOOL_DIR = DATA / "raw" / "tool"
OUT_DIR  = BASE / "outputs"


def stats(values):
    """(n, min, max, mean, population std) — matches the benchmark's own summary."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n > 1 else 0.0
    return n, min(values), max(values), mean, std


def fmt(s):
    if s is None:
        return "n=0"
    n, mn, mx, mean, std = s
    return f"n={n:<5} min={mn:>7.2f}  max={mx:>8.2f}  mean={mean:>7.2f} ± {std:<7.2f}"


def main():
    ap = argparse.ArgumentParser(description="Summarize replayed tool-call latency.")
    ap.add_argument("--iterations", type=int, default=3,
                    help="replays per call; a call with exactly this many rows is 'complete'")
    args = ap.parse_args()

    if not TOOL_DIR.is_dir():
        raise SystemExit(f"not found: {TOOL_DIR}")
    paths = sorted(p for p in TOOL_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no per-tool CSVs in {TOOL_DIR}")

    L = []
    L.append("=" * 78)
    L.append("  TOOL LATENCY  (replayed calls, seconds)")
    L.append(f"  {args.iterations} replays per call; scrape_and_extract_info excludes the summarizer LLM")
    L.append("=" * 78)

    grand_rows = grand_err = 0
    for path in paths:
        tool = path.stem
        all_times, err_times = [], []
        by_call = defaultdict(list)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    t = float(row["runtime_s"])
                except (KeyError, ValueError):
                    continue
                all_times.append(t)
                by_call[(row["filename"], row["turn"])].append(t)
                if row.get("output_20", "").startswith("ERR:"):
                    err_times.append(t)

        complete = [t for times in by_call.values()
                    if len(times) == args.iterations for t in times]
        grand_rows += len(all_times)
        grand_err += len(err_times)

        L.append("")
        L.append(f"  {tool}")
        L.append(f"  {'─' * 70}")
        if not all_times:
            L.append("    no rows")
            continue
        L.append(f"    calls {len(by_call):<5}      {fmt(stats(all_times))}")
        L.append(f"    complete only     {fmt(stats(complete))}")
        if len(complete) != len(all_times):
            L.append(f"    NOTE: {len(all_times) - len(complete)} rows belong to calls "
                     f"without all {args.iterations} replays")
    L.append("")
    L.append("=" * 78)
    L.append(f"  {grand_rows:,} rows across {len(paths)} tools"
             + (f"; {grand_err} ERR" if grand_err else "; no errors"))
    L.append("=" * 78)

    text = "\n".join(L)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "table2_miro.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
