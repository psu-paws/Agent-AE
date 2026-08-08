"""
Measure real tool latency by replaying the tool calls the GAIA runs actually made.

For each raw/tool/tool_calls/*_tools.csv, replay its target-tool calls in turn
order, and repeat the whole sequence NUM_ITERATIONS times. Sequence order matters:
create_sandbox returns a sandbox_id that later run_python_code calls must reuse, so
replaying tools in isolation would not reproduce the real latency.

Reads:  ../../../raw/tool/tool_calls/*_tools.csv   (turn, tool_name, args_json)
Writes: ../../../raw/tool/{tool_name}.csv          (one row per call per iteration)
        columns: filename, turn, iteration, args_json, runtime_s, output_20

Resumable and idempotent: a row already present for (filename, turn, iteration) is
skipped, so re-running against the shipped results replays nothing.

This script only measures. The latency summary lives in
mirothinker/analysis/tool_latency.py, which reads the same CSVs.

Usage (from apps/miroflow-agent/):
    uv run python3 benchmarks/benchmark_mcp_tools_full.py

Required env vars:
    SERPER_API_KEY  – google_search
    JINA_API_KEY    – scrape_and_extract_info. This benchmark pins the server to
                      scrape-only via BENCHMARK_SKIP_SUMMARY_LLM=1, so runtime_s
                      excludes the summarizer LLM. Real runs leave it unset.
    E2B_API_KEY     – create_sandbox / run_python_code
"""

import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from mcp import StdioServerParameters
from miroflow_tools.manager import ToolManager

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIRO_ROOT = Path(__file__).resolve().parents[4]      # GAIATrace/mirothinker
TOOL_DIR = MIRO_ROOT / "raw" / "tool"
FILES_DIR = TOOL_DIR / "tool_calls"
RESULTS_DIR = TOOL_DIR

# How many full-sequence replays per file.
NUM_ITERATIONS = 3

TOOL_SERVER = {
    "google_search": "tool-google-search",
    "scrape_and_extract_info": "jina_scrape_llm_summary",
    "create_sandbox": "tool-python",
    "run_python_code": "tool-python",
    "download_file_from_internet_to_sandbox": "tool-python",
}
TARGET_TOOLS = set(TOOL_SERVER)
OUTPUT_COLUMNS = ["filename", "turn", "iteration", "args_json", "runtime_s", "output_20"]


def _env(key, default=""):
    return os.environ.get(key, default)


SERVER_CONFIGS = [
    {
        "name": "tool-google-search",
        "params": StdioServerParameters(
            command=sys.executable,
            args=["-m", "miroflow_tools.mcp_servers.searching_google_mcp_server"],
            env={
                "SERPER_API_KEY": _env("SERPER_API_KEY"),
                "SERPER_BASE_URL": _env("SERPER_BASE_URL", "https://google.serper.dev"),
                "JINA_API_KEY": _env("JINA_API_KEY"),
                "JINA_BASE_URL": _env("JINA_BASE_URL", "https://r.jina.ai"),
            },
        ),
    },
    {
        "name": "jina_scrape_llm_summary",
        "params": StdioServerParameters(
            command=sys.executable,
            args=["-m", "miroflow_tools.dev_mcp_servers.jina_scrape_llm_summary"],
            env={
                "JINA_API_KEY": _env("JINA_API_KEY"),
                "JINA_BASE_URL": _env("JINA_BASE_URL", "https://r.jina.ai"),
                # This benchmark always measures scraping only.
                "BENCHMARK_SKIP_SUMMARY_LLM": "1",
            },
        ),
    },
    {
        "name": "tool-python",
        "params": StdioServerParameters(
            command=sys.executable,
            args=["-m", "miroflow_tools.mcp_servers.python_mcp_server"],
            env={"E2B_API_KEY": _env("E2B_API_KEY")},
        ),
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_file_calls(csv_path):
    """Target-tool rows from one *_tools.csv, in turn order."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["tool_name"] in TARGET_TOOLS]


def parse_sandbox_id(result_str):
    m = re.search(r"sandbox_id[:\s]+([a-zA-Z0-9_-]+)", result_str)
    return m.group(1) if m else None


def load_completed(tool_name):
    """(filename, turn, iteration) triples already recorded for one tool."""
    path = RESULTS_DIR / f"{tool_name}.csv"
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["filename"], r["turn"], r["iteration"]) for r in csv.DictReader(f)}


def open_writers():
    """One append-mode CSV writer per target tool."""
    writers = {}
    for tool in TARGET_TOOLS:
        path = RESULTS_DIR / f"{tool}.csv"
        is_new = not path.exists()
        fh = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        if is_new:
            w.writeheader()
        writers[tool] = (fh, w)
    return writers


async def timed_call(tm, tool_name, args):
    """Run one tool call, returning (seconds, first 200 chars of result, result_str)."""
    t0 = time.perf_counter()
    try:
        result_str = str(await tm.execute_tool_call(TOOL_SERVER[tool_name], tool_name, args))
        return time.perf_counter() - t0, result_str[:200].replace("\n", " "), result_str
    except Exception as exc:
        # Recorded as a row so the replay stays aligned; note that analyze() counts
        # these ERR rows in the latency stats.
        return time.perf_counter() - t0, f"ERR:{str(exc)[:194]}", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark():
    if not FILES_DIR.is_dir():
        raise SystemExit(f"input dir not found: {FILES_DIR}")
    csv_files = sorted(FILES_DIR.glob("*_tools.csv"))
    if not csv_files:
        raise SystemExit(f"no *_tools.csv in {FILES_DIR}")
    print(f"Found {len(csv_files)} tool CSV files in {FILES_DIR}")
    print(f"Writing results to {RESULTS_DIR}")

    completed = {t: load_completed(t) for t in TARGET_TOOLS}
    writers = open_writers()
    tm = ToolManager(SERVER_CONFIGS)

    try:
        for file_idx, csv_path in enumerate(csv_files, 1):
            filename = csv_path.name
            calls = load_file_calls(csv_path)
            if not calls:
                continue
            print(f"\n[{file_idx}/{len(csv_files)}] {filename}  ({len(calls)} target calls)")

            for iteration in range(1, NUM_ITERATIONS + 1):
                sandbox_id = None       # live sandbox for this replay of this file

                for call in calls:
                    tool_name, turn, args_json = call["tool_name"], call["turn"], call["args_json"]

                    if (filename, turn, str(iteration)) in completed[tool_name]:
                        print(f"  skip  iter={iteration} turn={turn} {tool_name}")
                        continue

                    try:
                        args = json.loads(args_json) if args_json.strip() else {}
                    except json.JSONDecodeError:
                        args = {}

                    # Sandbox tools need the id from this replay's create_sandbox. If that
                    # call was skipped as already-done, there is no live id: run_python_code
                    # falls back to "default" (and will fail), download spins up a fresh one.
                    if tool_name == "run_python_code":
                        args["sandbox_id"] = sandbox_id or "default"
                    elif tool_name == "download_file_from_internet_to_sandbox":
                        if not sandbox_id:
                            fresh = await tm.execute_tool_call("tool-python", "create_sandbox", {})
                            sandbox_id = parse_sandbox_id(str(fresh))
                        args["sandbox_id"] = sandbox_id

                    runtime_s, output_20, result_str = await timed_call(tm, tool_name, args)

                    if tool_name == "create_sandbox":
                        sandbox_id = parse_sandbox_id(result_str) or sandbox_id

                    fh, w = writers[tool_name]
                    w.writerow({
                        "filename": filename,
                        "turn": turn,
                        "iteration": iteration,
                        "args_json": args_json,
                        "runtime_s": f"{runtime_s:.4f}",
                        "output_20": output_20,
                    })
                    fh.flush()
                    completed[tool_name].add((filename, turn, str(iteration)))

                    print(f"  iter={iteration} turn={turn:>3} {tool_name:<25} "
                          f"{runtime_s:.2f}s  {output_20!r}")
    finally:
        for fh, _ in writers.values():
            fh.close()

    print(f"\nDone. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
