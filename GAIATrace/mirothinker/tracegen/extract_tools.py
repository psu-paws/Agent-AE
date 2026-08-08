"""
extract_tools.py

For each transcript in traces/transcripts/, extracts tool calls from <use_mcp_tool>
blocks and writes one raw/tool/tool_calls/{stem}_tools.csv per task.

Those CSVs drive the tool benchmark
(agent/apps/miroflow-agent/benchmarks/benchmark_mcp_tools.py), whose measured
latencies traces/merge.py turns into the inter_request_latency column.

Columns: turn, tool_name, args_json   (turn is 0-based)
"""

import csv
import json
import re
from pathlib import Path

BASE          = Path(__file__).parent                        # mirothinker/tracegen
DATA          = BASE.parent                                  # mirothinker
RAW_DIR       = DATA / "traces" / "transcripts"                # input: parser.py transcripts
TOOLS_DIR     = DATA / "raw" / "tool" / "tool_calls"          # output
TOOLS_CSV_FMT = "{stem}_tools.csv"
TOOLS_FIELDNAMES = ["turn", "tool_name", "args_json"]

# Log markers — change these, not the code below.
TURN_RE    = re.compile(r'--- Turn (\d+)/\d+[^-]*---')
MCP_RE     = re.compile(r'<use_mcp_tool>(.*?)</use_mcp_tool>', re.DOTALL)
TOOL_RE    = re.compile(r'<tool_name>(.*?)</tool_name>', re.DOTALL)
ARGS_RE    = re.compile(r'<arguments>(.*?)</arguments>', re.DOTALL)


def extract_from_file(path: Path) -> list[dict]:
    txt = path.read_text(encoding="utf-8", errors="replace")

    # Split into (turn_number, chunk) pairs; chunk 0 = preamble (skip)
    parts = TURN_RE.split(txt)
    # parts = [preamble, turn_num, turn_body, turn_num, turn_body, ...]

    rows = []
    i = 1  # skip preamble at index 0
    while i + 1 < len(parts):
        turn_num = int(parts[i])
        body     = parts[i + 1]
        i += 2

        for block in MCP_RE.finditer(body):
            content   = block.group(1)
            tool_match = TOOL_RE.search(content)
            args_match = ARGS_RE.search(content)
            if not tool_match or not args_match:
                continue
            tool_name = tool_match.group(1).strip()
            args_raw  = args_match.group(1).strip()
            try:
                args_json = json.dumps(json.loads(args_raw), sort_keys=True)
            except json.JSONDecodeError:
                args_json = args_raw
            rows.append({"turn": turn_num - 1, "tool_name": tool_name, "args_json": args_json})

    return rows


def main():
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(RAW_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} txt files")

    from collections import Counter
    total_counts = Counter()

    for path in txt_files:
        rows = extract_from_file(path)
        out  = TOOLS_DIR / TOOLS_CSV_FMT.format(stem=path.stem)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TOOLS_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        for r in rows:
            total_counts[r["tool_name"]] += 1
        print(f"  {path.name}: {len(rows)} tool calls → {out.name}")

    print(f"\nTotal tool calls across all files: {sum(total_counts.values())}")
    for tool, n in total_counts.most_common():
        print(f"  {n:4d}  {tool}")


if __name__ == "__main__":
    main()
