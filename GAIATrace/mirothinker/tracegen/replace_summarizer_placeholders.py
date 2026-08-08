"""
replace_summarizer_placeholders.py — fill in the summarizer rows of the traces.

The summarizer is gpt-4o-mini behind the OpenAI API, which reports token counts but
not token ids. The ids are reconstructed by re-rendering each recorded request
(raw/summarizer/*.json, named by the row's `sum_io_basename`) through the
o200k_harmony chat template. Decode matches the API's count exactly; prefill lands
within ~0.5% of it, since OpenAI's server-side framing is not published.

Rows are rewritten in place and their row_kind flipped to 'summarizer', so re-runs
are no-ops unless --force. Each CSV is written atomically.

Usage:
  python replace_summarizer_placeholders.py [trace_dir] [--summarizer-io PATH]
                                            [--force] [--verify]
"""

import argparse
import ast
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import jinja2
import tiktoken

csv.field_size_limit(sys.maxsize)

_SCRIPT_DIR    = Path(__file__).parent                     # mirothinker/tracegen
DATA           = _SCRIPT_DIR.parent                        # mirothinker
TRACES_DIR     = DATA / "traces" / "session_traces"        # CSVs to patch in place
SUMMARIZER_DIR = DATA / "raw" / "summarizer"               # gpt-4o-mini request/response dumps

# ── Tokenizer + template setup ─────────────────────────────────────────────────

def _raise(msg):
    raise ValueError(f"Template Error: {msg}")

def _build_env():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_SCRIPT_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals['raise_exception'] = _raise
    return env

_env = _build_env()
_template = _env.get_template('chat_template_4o.jinja')
_encoding = tiktoken.get_encoding('o200k_harmony')


def tokenize_summarizer(messages: list, assistant_text: str) -> dict:
    """
    Render prefill from messages, encode assistant_text as decode.
    Returns {'num_prefill_tokens', 'num_decode_tokens', 'tokens'}.

    chat_template_4o.jinja only sets loop_messages when the first message is
    system/developer. Summarizer calls have no system prompt, so we prepend an
    empty system message — the template renders it as an empty developer block
    and correctly iterates over the remaining user messages.
    """
    if not messages or messages[0].get('role') not in ('system', 'developer'):
        messages = [{'role': 'system', 'content': ''}] + list(messages)

    rendered = _template.render(
        messages=messages,
        tools=None,
        builtin_tools=[],
        add_generation_prompt=True,
    )
    prefill_ids = _encoding.encode(rendered, allowed_special='all')
    decode_ids  = _encoding.encode(assistant_text, allowed_special='all')
    return {
        'num_prefill_tokens': len(prefill_ids),
        'num_decode_tokens':  len(decode_ids),
        'tokens':             str(prefill_ids + decode_ids),
    }


# ── Per-file processing ────────────────────────────────────────────────────────

PLACEHOLDER  = 'summarizer_placeholder'
REPLACED     = 'summarizer'
VERIFY_ROWS  = 3    # --verify prints the shortest N rows, where diffs are readable

def _needs_replace(row: dict, force: bool) -> bool:
    if row.get('row_kind') == PLACEHOLDER:
        return True
    if force and row.get('row_kind') == REPLACED:
        return True
    return False

def process_csv(csv_path: Path, io_dir: Path, force: bool = False) -> tuple[int, int]:
    """
    Returns (n_replaced, n_missing).
    """
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    if not rows:
        return 0, 0

    fieldnames = list(rows[0].keys())
    n_replaced = n_missing = n_error = 0

    for row in rows:
        if not _needs_replace(row, force):
            continue

        basename = row.get('sum_io_basename', '').strip()
        if not basename:
            print(f'  SKIP: missing sum_io_basename in {csv_path.name}')
            n_missing += 1
            continue

        json_path = io_dir / basename
        if not json_path.exists():
            print(f'  MISSING: {json_path}')
            n_missing += 1
            continue

        try:
            data = json.load(open(json_path, encoding='utf-8'))
            payload  = data.get('request_payload', {})
            messages = payload.get('messages', [])
            asst     = data.get('assistant_text', '')

            if not messages:
                print(f'  WARN: no messages in {basename}')
                n_missing += 1
                continue

            tok = tokenize_summarizer(messages, asst)

            stored_p = int(data.get('prompt_tokens', 0))
            stored_d = int(data.get('completion_tokens', 0))
            print(f'  {basename}')
            print(f'    prefill : computed={tok["num_prefill_tokens"]:>6}  gt={stored_p:>6}  ratio={tok["num_prefill_tokens"]/stored_p:.3f}' if stored_p else f'    prefill : computed={tok["num_prefill_tokens"]:>6}  gt=N/A')
            print(f'    decode  : computed={tok["num_decode_tokens"]:>6}  gt={stored_d:>6}  ratio={tok["num_decode_tokens"]/stored_d:.3f}' if stored_d else f'    decode  : computed={tok["num_decode_tokens"]:>6}  gt=N/A')

            row['row_kind']           = REPLACED
            row['num_prefill_tokens'] = tok['num_prefill_tokens']
            row['num_decode_tokens']  = tok['num_decode_tokens']
            row['tokens']             = tok['tokens']
            n_replaced += 1

        except Exception as e:
            print(f'  ERROR {basename}: {e}')
            n_error += 1

    # Atomic write via temp file
    tmp = csv_path.with_suffix('.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)

    return n_replaced, n_missing


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Replace summarizer_placeholder rows with real tokens.')
    parser.add_argument('trace_dir', nargs='?', default=str(TRACES_DIR),
                        help=f'Directory containing the trace CSVs (default: {TRACES_DIR})')
    parser.add_argument('--summarizer-io', default=str(SUMMARIZER_DIR),
                        help=f'Path to the summarizer JSON dumps (default: {SUMMARIZER_DIR})')
    parser.add_argument('--force', action='store_true',
                        help='Re-replace already-replaced summarizer rows (allows re-runs after fixes)')
    parser.add_argument('--verify', action='store_true',
                        help='After replacing, decode the ids back to text and print them '
                             'beside the recorded request/response')
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    io_dir    = Path(args.summarizer_io)

    if not trace_dir.is_dir():
        sys.exit(f'Error: trace_dir not found: {trace_dir}')
    if not io_dir.is_dir():
        sys.exit(f'Error: summarizer_io not found: {io_dir}')

    csvs = sorted(trace_dir.glob('*.csv'))
    print(f'Found {len(csvs)} CSV files in {trace_dir}')
    print(f'summarizer_io: {io_dir}')
    print()

    total_replaced = total_missing = 0
    for csv_path in csvs:
        replaced, missing = process_csv(csv_path, io_dir, args.force)
        if replaced or missing:
            print(f'  [OK] {csv_path.name}: replaced={replaced}  missing={missing}')
        total_replaced += replaced
        total_missing  += missing

    print()
    print(f'Done. Total replaced={total_replaced}  missing/error={total_missing}')

    if args.verify:
        print()
        print('=== VERIFICATION: shortest summarizer row, full text comparison ===')
        # Collect all replaced summarizer rows across all CSVs
        candidates = []
        for csv_path in csvs:
            for row in csv.DictReader(open(csv_path, encoding='utf-8')):
                if row.get('row_kind') != REPLACED:
                    continue
                basename = row.get('sum_io_basename', '').strip()
                if not (io_dir / basename).exists():
                    continue
                candidates.append((int(row['num_prefill_tokens']), csv_path, row))

        if not candidates:
            print('  No replaced summarizer rows found.')
        else:
            # Pick the shortest prefill (easiest to diff)
            candidates.sort(key=lambda x: x[0])
            for _ , csv_path, row in candidates[:VERIFY_ROWS]:
                basename  = row['sum_io_basename'].strip()
                data      = json.load(open(io_dir / basename, encoding='utf-8'))
                gt_input  = data['request_payload']['messages'][0]['content']
                gt_output = data.get('assistant_text', '')
                stored_p  = int(data.get('prompt_tokens', 0))
                stored_d  = int(data.get('completion_tokens', 0))
                ids       = ast.literal_eval(row['tokens'])
                prefill_n = int(row['num_prefill_tokens'])
                prefill_text = _encoding.decode(ids[:prefill_n])
                decode_text  = _encoding.decode(ids[prefill_n:])

                print(f'\n{"="*70}')
                print(f'File:    {csv_path.name} / {basename}')
                print(f'Prefill: computed={prefill_n}  gt={stored_p}  diff={prefill_n - stored_p:+d}')
                print(f'Decode:  computed={int(row["num_decode_tokens"])}  gt={stored_d}  diff={int(row["num_decode_tokens"]) - stored_d:+d}')
                print(f'\n--- GT INPUT (full) ---')
                print(gt_input)
                print(f'\n--- DECODED PREFILL (full) ---')
                print(prefill_text)
                print(f'\n--- GT OUTPUT (full) ---')
                print(gt_output)
                print(f'\n--- DECODED DECODE (full) ---')
                print(decode_text)


if __name__ == '__main__':
    main()
