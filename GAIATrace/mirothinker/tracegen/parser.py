"""
parser.py — MiroFlow run logs → per-request trace CSVs.

Counterpart of owl/tracegen/parser.py. Reads one MiroFlow JSON log per task from
raw/agent/, rebuilds the prompt the inference server actually saw for each turn,
tokenizes it, and writes:

    traces/session_traces/<task_id>.csv   one row per request
    traces/transcripts/<task_id>.txt      human-readable rendering of the same log

Main-model rows are complete. Summarizer calls (gpt-4o-mini) appear only as
token counts in the log, so they are emitted as `summarizer_placeholder` rows
carrying the hints needed to find their recorded I/O; replace_summarizer_placeholders.py
fills them from raw/summarizer/.

Usage:
    python parser.py                 (needs the Qwen3 tokenizer; downloads on first run)
"""

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# ==========================================
# 0. CONFIGURATION
# ==========================================

_SCRIPT_DIR   = Path(__file__).resolve().parent          # mirothinker/tracegen
DATA          = _SCRIPT_DIR.parent                       # mirothinker
RAW_DIR       = DATA / "raw" / "agent"                   # MiroFlow JSON logs
TRACES_DIR    = DATA / "traces" / "session_traces"       # per-request CSVs
TRANSCRIPT_DIR = DATA / "traces" / "transcripts"         # readable rendering

TOKENIZER_NAME = "Qwen/Qwen3-8B"

# Log markers ----------------------------------------------------------------
# step_logs carry CUMULATIVE totals; differencing them gives per-turn counts,
# which are cross-checked against the tokenizer below.
TOKEN_RE = re.compile(r"Input:\s*(\d+),\s*Output:\s*(\d+)")
# Filenames look like task_<uuid>_attempt-N_format-retry-M_<timestamp>.json
TASK_RE  = re.compile(r"task_([0-9a-f-]+)_attempt-\d+_format-retry-(\d+)_")

# Text written in place of a dropped tool result (mirrors the agent's own
# base_client._remove_tool_result_from_messages).
RETENTION_PLACEHOLDER = "Tool result is omitted to save tokens."

# Columns describing the summarizer call a placeholder row stands for.
SUM_COLUMNS = [
    "sum_url", "sum_prompt_tokens", "sum_completion_tokens", "sum_model_used",
    "sum_io_basename", "sum_io_prompt_digest", "sum_openai_response_id",
    "sum_system_fingerprint",
]
FIELDNAMES = ["row_kind", "num_prefill_tokens", "num_decode_tokens", "tokens"] + SUM_COLUMNS

KIND_MAIN        = "main"
KIND_PLACEHOLDER = "summarizer_placeholder"

# Transcript truncation (readability only; never used for tokenization).
TXT_SYSTEM_CHARS = 2000
TXT_USER_CHARS   = 3000


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


# ==========================================
# 1. PROMPT RECONSTRUCTION
# ==========================================

def build_chatml_prefill(system_prompt, messages):
    """ChatML prompt up to the point the assistant starts generating."""
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_chatml_decode(assistant_content):
    return f"{assistant_content}<|im_end|>"


def apply_retention_policy(messages_with_sys, keep_tool_result):
    """Blank out all but the newest `keep_tool_result` tool results.

    The first user message (the task itself) is always kept. -1 keeps everything.
    This is what makes prefill grow sub-linearly, so it has to be replayed exactly.
    """
    msgs = [m.copy() for m in messages_with_sys]
    if keep_tool_result == -1:
        return msgs

    user_indices = [i for i, m in enumerate(msgs) if m.get("role") in ("user", "tool")]
    if len(user_indices) <= 1:
        return msgs

    tool_result_indices = user_indices[1:]
    num_keep = min(keep_tool_result, len(tool_result_indices)) if keep_tool_result > 0 else 0
    keep = set([user_indices[0]] + (tool_result_indices[-num_keep:] if num_keep else []))

    for i, msg in enumerate(msgs):
        if msg.get("role") in ("user", "tool") and i not in keep:
            msg["content"] = RETENTION_PLACEHOLDER
    return msgs


def extract_logged_token_counts(step_logs, num_turns):
    """Per-turn (input, output) from the log's cumulative counters.

    Returns None when the number of counter lines does not equal the turn count,
    which means the run had retries and the two cannot be aligned.
    """
    cumulative = []
    for s in step_logs:
        m = TOKEN_RE.match(s.get("message", ""))
        if m:
            cumulative.append((int(m.group(1)), int(m.group(2))))
    if len(cumulative) != num_turns:
        return None

    per_turn = []
    prev_in = prev_out = 0
    for cum_in, cum_out in cumulative:
        per_turn.append((cum_in - prev_in, cum_out - prev_out))
        prev_in, prev_out = cum_in, cum_out
    return per_turn


# ==========================================
# 2. SUMMARIZER PLACEHOLDERS
# ==========================================

def _empty_sum_hints():
    return {k: "" for k in SUM_COLUMNS}


def is_summarizer_message(msg):
    """True when a user message is the JSON tool result of a summarizer call."""
    raw = msg.get("content") or ""
    if not isinstance(raw, str) or not raw.strip().startswith("{"):
        return False
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return (isinstance(obj, dict)
            and obj.get("prompt_tokens") is not None
            and obj.get("completion_tokens") is not None)


def parse_summarizer_message(msg):
    if not is_summarizer_message(msg):
        return None
    obj = json.loads((msg.get("content") or "").strip())
    try:
        return {"url": (obj.get("url") or "").strip(),
                "prompt_tokens": int(obj["prompt_tokens"]),
                "completion_tokens": int(obj["completion_tokens"]),
                "model_used": str(obj.get("model_used") or "")}
    except (KeyError, TypeError, ValueError):
        return None


def load_summarizer_io_pool(summ_dir):
    """Recorded gpt-4o-mini calls, ordered by the timestamp in their filename."""
    pool = []
    if not summ_dir.is_dir():
        return pool
    for p in summ_dir.glob("*.json"):
        if not p.stem or p.stem == "summarizer_calls":
            continue
        try:
            sort_ms = int(p.stem.split("_", 1)[0])
        except ValueError:
            sort_ms = 0
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rec, dict):
            pool.append((sort_ms, p.name, rec, str(p.resolve())))
    pool.sort(key=lambda x: (x[0], x[1]))
    return pool


def take_matching_io_record(pool, used, url, prompt_tokens, completion_tokens):
    """First unused record matching url + both token counts, else url alone."""
    def find(pred):
        for item in pool:
            _, fname, rec, path = item
            if path not in used and pred(rec, fname):
                return item
        return None

    hit = find(lambda rec, _: rec.get("url") == url
               and int(rec.get("prompt_tokens") or -1) == prompt_tokens
               and int(rec.get("completion_tokens") or -1) == completion_tokens)
    return hit or find(lambda rec, _: rec.get("url") == url)


def io_record_hints(fname, rec):
    """Enough to open the right raw/summarizer file without a task-level index."""
    stem = Path(fname).stem
    rj = rec.get("response_json")
    rj = rj if isinstance(rj, dict) else {}
    return {"sum_io_basename": fname,
            "sum_io_prompt_digest": stem.split("_", 1)[1] if "_" in stem else "",
            "sum_openai_response_id": str(rj.get("id") or ""),
            "sum_system_fingerprint": str(rj.get("system_fingerprint") or "")}


def interleave_placeholders(main_rows, msgs, num_turns, io_pool):
    """One main row per turn, plus a placeholder wherever a summarizer call ran.

    A summarizer call shows up as the user message feeding the next turn, i.e. at
    msgs[2], msgs[4], … — so the row order stays main → summarizer → main → …
    """
    used, unmatched, out = set(), 0, []
    for k in range(num_turns):
        row = {"row_kind": KIND_MAIN, **main_rows[k], **_empty_sum_hints()}
        out.append(row)
        if k >= num_turns - 1:
            break

        user_idx = 2 * (k + 1)
        if user_idx >= len(msgs):
            continue
        parsed = parse_summarizer_message(msgs[user_idx])
        if not parsed:
            continue

        hints = {**_empty_sum_hints(),
                 "sum_url": parsed["url"],
                 "sum_prompt_tokens": str(parsed["prompt_tokens"]),
                 "sum_completion_tokens": str(parsed["completion_tokens"]),
                 "sum_model_used": parsed["model_used"]}
        if io_pool:
            hit = take_matching_io_record(io_pool, used, parsed["url"],
                                          parsed["prompt_tokens"], parsed["completion_tokens"])
            if hit:
                used.add(hit[3])
                hints.update(io_record_hints(hit[1], hit[2]))
            else:
                unmatched += 1

        out.append({"row_kind": KIND_PLACEHOLDER, "num_prefill_tokens": -1,
                    "num_decode_tokens": -1, "tokens": "[]", **hints})
    return out, unmatched


# ==========================================
# 3. PER-TASK PROCESSING
# ==========================================

def process_trace(trace_path, tokenizer, csv_path, io_pool):
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)

    mh = data.get("main_agent_message_history", {})
    if isinstance(mh, list):
        print(f"  SKIP {trace_path.name}: message history is a list (unexpected format)")
        return None

    system_prompt = mh.get("system_prompt", "")
    msgs = mh.get("message_history", [])
    num_turns = len(msgs) // 2
    if not msgs or num_turns == 0:
        print(f"  SKIP {trace_path.name}: no complete turns")
        return None

    keep_tool_result = data.get("env_info", {}).get("keep_tool_result", -1)

    main_rows = []
    for k in range(1, num_turns + 1):
        # Everything the model saw before it answered turn k.
        with_sys = [{"role": "system", "content": system_prompt}] + msgs[: 2 * k - 1]
        filtered = apply_retention_policy(with_sys, keep_tool_result)

        prefill_ids = tokenizer.encode(
            build_chatml_prefill(filtered[0]["content"], filtered[1:]),
            add_special_tokens=False)
        decode_ids = tokenizer.encode(
            build_chatml_decode(msgs[2 * k - 1]["content"]),
            add_special_tokens=False)

        main_rows.append({"num_prefill_tokens": len(prefill_ids),
                          "num_decode_tokens": len(decode_ids),
                          "tokens": str(prefill_ids + decode_ids)})

    rows, unmatched = interleave_placeholders(main_rows, msgs, num_turns, io_pool)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logged = extract_logged_token_counts(data.get("step_logs", []), num_turns)
    return main_rows, logged, unmatched


def write_transcript(trace_path, txt_path):
    """Readable rendering of one run. Long fields are truncated — this file is for
    humans and for score_gaia.py's \\boxed{} lookup, never for tokenization."""
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)

    mh = data.get("main_agent_message_history", {})
    if isinstance(mh, list):
        return
    system_prompt = mh.get("system_prompt", "")
    msgs = mh.get("message_history", [])
    if not msgs:
        return

    env = data.get("env_info", {})
    num_turns = len(msgs) // 2
    logged = extract_logged_token_counts(data.get("step_logs", []), num_turns)

    L = ["=" * 70,
         f"Task ID:          {data.get('task_id', trace_path.stem)}",
         f"Model:            {env.get('llm_model_name', 'unknown')}",
         f"LLM Turns:        {num_turns}",
         f"Keep Tool Result: {env.get('keep_tool_result', -1)}",
         f"Max Turns:        {env.get('main_agent_max_turns', '?')}",
         "=" * 70 + "\n",
         f"[SYSTEM PROMPT] ({len(system_prompt)} chars)",
         system_prompt[:TXT_SYSTEM_CHARS]]
    if len(system_prompt) > TXT_SYSTEM_CHARS:
        L.append(f"... ({len(system_prompt) - TXT_SYSTEM_CHARS} more chars truncated)")
    L.append("")

    for k in range(1, num_turns + 1):
        user_msg, asst_msg = msgs[2 * k - 2], msgs[2 * k - 1]
        counts = ""
        if logged and k - 1 < len(logged):
            lp, ld = logged[k - 1]
            counts = f", prefill={lp}, decode={ld}"
        L.append(f"--- Turn {k}/{num_turns}{counts} ---")

        label = "Task" if k == 1 else "Tool Result"
        user_content = user_msg.get("content", "")
        if k == num_turns and "Summarize the above conversation" in user_content:
            label = "Summary Prompt"
        L.append(f"\n[USER - {label}] ({len(user_content)} chars)")
        L.append(user_content[:TXT_USER_CHARS])
        if len(user_content) > TXT_USER_CHARS:
            L.append(f"... ({len(user_content) - TXT_USER_CHARS} more chars truncated)")

        asst_content = asst_msg.get("content", "")
        L.append(f"\n[ASSISTANT] ({len(asst_content)} chars)")
        L.append(asst_content)
        L.append("")

    tool_calls = [s for s in data.get("step_logs", []) if "Tool Call Start" in s.get("step_name", "")]
    if tool_calls:
        L += ["\n" + "=" * 70, f"TOOL CALLS ({len(tool_calls)} total)", "=" * 70]
        L += [f"  {tc.get('message', '')}" for tc in tool_calls]

    txt_path.write_text("\n".join(L), encoding="utf-8")


# ==========================================
# 4. VERIFICATION
# ==========================================

def check_against_log(main_rows, logged, task_id):
    """Compare our token counts against the counts the server reported.

    `logged` is None when retries made the counters unalignable — that is a
    property of the run, not an error, so it is reported separately.
    """
    if logged is None:
        return "skipped"
    ok = True
    for i in range(min(len(main_rows), len(logged))):
        p, d = main_rows[i]["num_prefill_tokens"], main_rows[i]["num_decode_tokens"]
        lp, ld = logged[i]
        if abs(p - lp) / max(lp, 1) > 0.001 or abs(d - ld) / max(ld, 1) > 0.001:
            print(f"  [{task_id}] turn {i+1}: prefill {p} vs logged {lp}, "
                  f"decode {d} vs logged {ld}")
            ok = False
    return "match" if ok else "mismatch"


def verify_csv(traces_dir):
    """Re-read every CSV: main rows must satisfy prefill+decode == len(tokens),
    placeholders must be exactly -1, -1, []."""
    print("\n=== RUNNING VERIFICATION ===")
    verified = errors = 0
    for csv_file in sorted(traces_dir.glob("*.csv")):
        with open(csv_file, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                p = int(row["num_prefill_tokens"])
                d = int(row["num_decode_tokens"])
                n = len(ast.literal_eval(row["tokens"]))
                if row.get("row_kind") == KIND_PLACEHOLDER:
                    bad = (p, d, n) != (-1, -1, 0)
                else:
                    bad = p + d != n
                if bad:
                    print(f"  {csv_file.name} row {i} MISMATCH: {p}+{d} vs len={n}")
                    errors += 1
                else:
                    verified += 1
    print(f"Verified {verified} rows correctly.")
    print("SUCCESS: All row counts match perfectly." if errors == 0
          else f"FAILURE: {errors} rows had mismatches.")


def deduplicate_traces(json_files):
    """One file per task: the highest format-retry."""
    best = {}
    for p in json_files:
        m = TASK_RE.search(p.name)
        if not m:
            continue
        tid, retry = m.group(1), int(m.group(2))
        if tid not in best or retry > best[tid][1]:
            best[tid] = (p, retry)
    return [v[0] for v in best.values()]


# ==========================================
# 5. MAIN
# ==========================================

def main():
    ap = argparse.ArgumentParser(description="MiroFlow run logs → per-request trace CSVs")
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="dir of MiroFlow JSON logs")
    ap.add_argument("--summarizer-io", type=Path, default=DATA / "raw" / "summarizer",
                    help="dir of recorded gpt-4o-mini calls")
    args = ap.parse_args()

    if not args.raw.is_dir():
        raise SystemExit(f"not found: {args.raw}")

    json_files = deduplicate_traces(sorted(args.raw.glob("*.json")))
    if not json_files:
        raise SystemExit(f"no MiroFlow JSON logs in {args.raw}")
    print(f"Found {len(json_files)} tasks in {args.raw}")

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {TOKENIZER_NAME} tokenizer …")
    tokenizer = load_tokenizer()
    io_pool = load_summarizer_io_pool(args.summarizer_io)
    print(f"Summarizer I/O records: {len(io_pool)}\n")

    tally = {"match": 0, "skipped": 0, "mismatch": 0}
    tasks = turns = unmatched_total = 0

    for trace_path in sorted(json_files):
        m = TASK_RE.search(trace_path.name)
        task_id = m.group(1) if m else trace_path.stem

        result = process_trace(trace_path, tokenizer,
                               TRACES_DIR / f"{task_id}.csv", io_pool)
        if result is None:
            continue
        write_transcript(trace_path, TRANSCRIPT_DIR / f"{task_id}.txt")

        main_rows, logged, unmatched = result
        tasks += 1
        turns += len(main_rows)
        unmatched_total += unmatched
        if unmatched:
            print(f"  [{task_id[:8]}] WARNING: {unmatched} summarizer call(s) had no "
                  f"recorded I/O; sum_* hints from the log are still present")
        tally[check_against_log(main_rows, logged, task_id[:8])] += 1

    print(f"\nProcessed {tasks} tasks, {turns} main-model turns.")
    print(f"  CSVs        : {TRACES_DIR}")
    print(f"  transcripts : {TRANSCRIPT_DIR}")
    if unmatched_total:
        print(f"  summarizer calls without recorded I/O: {unmatched_total}")
    print(f"  token counts vs server log: {tally['match']} match, "
          f"{tally['mismatch']} mismatch, {tally['skipped']} unalignable (retries)")

    verify_csv(TRACES_DIR)


if __name__ == "__main__":
    main()
