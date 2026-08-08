import jinja2
import json
import tiktoken
import ast
import re
import csv
import sys
import os
import bisect
from pathlib import Path

csv.field_size_limit(sys.maxsize)   # token lists exceed the default 131072-char field limit

# ==========================================
# 0. CONFIGURATION
# ==========================================

# --- Paths ---------------------------------------------------------
RAW_SUBDIR    = ('raw', 'agent')              # OWL stdout logs
OUT_SUBDIR    = ('traces', 'session_traces')      # generated per-request CSVs
TOOLS_SUBDIR  = ('raw', 'tool', 'tool_calls')      # tool calls per request; drives tool_bench.py
RAW_GLOB      = 'L*.txt'
TRACE_CSV_FMT = '{stem}.csv'
TOOLS_CSV_FMT = '{stem}_tools.csv'

# --- Log markers ------------------------------------------------------------
REQUEST_IN_RE    = re.compile(r'Request IN:\s+\[')
ANSWER_OUT_RE    = re.compile(r'Answer OUT\d+:\s*(?:(\d+)\s+)?(gpt-4o|openai/gpt-oss)')
CHAT_MSG_MARKER  = 'ChatCompletionMessage'
OSS_LIST_MARKER  = "['<|channel|>'"
COMMENT_PREFIX   = '#'
FUNCTION_CALL_RE = re.compile(r"Function\(arguments='(.*?)',\s*name='(\w+)'", re.DOTALL)
ELAPSED_TOOL_RE  = re.compile(r'Elapsed Time Tool Call Async \(sec\):\s*([\d.]+)')
ELAPSED_CODE_RE  = re.compile(r'Code Run Execution \(sec\):\s*([\d.]+)')

# --- Models / templates / tokenizer -----------------------------------------
MODEL_4O_ID    = 'gpt-4o-2024-08-06'
MODEL_OSS_ID   = 'openai/gpt-oss-120b'
TEMPLATE_4O    = 'chat_template_4o.jinja'   # the only template rendered
TOKENIZER_NAME = 'o200k_harmony'

# Images are never tokenized: each inline image is swapped for IMG_SENTINEL and
# replaced by IMAGE_PLACEHOLDER_TOKENS zero-ids at its exact position, so the
# base64 payload never reaches the tokenizer.
IMAGE_PLACEHOLDER_TOKENS = 1105          # empirical, typical browser screenshot
IMG_SENTINEL             = "\x01\x02IMG_PH\x02\x01"
IMAGE_DATA_URI_PREFIX    = 'data:image/'
IMAGE_URL_PLACEHOLDER    = '[IMAGE]'

# --- Row sentinels (shared with replace_oss_rows.py) ------------------------
AGENT_OSS_UNRESOLVED = 9     # "other": OSS role not identified by replace_oss_rows.py

SENTINEL_ASYNC_OSS  = 'ASYNC_OSS'
SENTINEL_SEQ_OSS    = 'SEQ_OSS'
SENTINEL_MISSING_4O = 'MISSING_4O'

# --- CSV columns ------------------------------------------------------------
TRACE_FIELDNAMES = ['num_prefill_tokens', 'num_decode_tokens', 'agent', 'dep', 'tokens']
TOOLS_FIELDNAMES = ['request_idx', 'tool_name', 'args_json', 'observed_elapsed']

# --- Agent system-prompt markers --------------------------------------------
# Only gpt-4o rows are labelled here; OSS labels come from replace_oss_rows.py.
M_PLANNER        = 'compose and decompose'
M_SEARCH         = 'search the web'
M_BROWSER_ACT_A  = 'helpful web agent'
M_BROWSER_ACT_B  = 'browsing the web'
M_ANSWERER       = 'answer questions and provide final answers'
M_DOC            = 'process documents and multimodal data'

TOOL_NAMES = {
    'extract_document_content', 'ask_question_about_image', 'ask_question_about_audio',
    'ask_question_about_video', 'execute_code', 'search_google', 'search_wiki',
    'search_wiki_revisions', 'search_archived_webpage', 'extract_excel_content',
}

# ==========================================
# 1. TOOL DEFINITIONS
# ==========================================
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tool_schemas.json'), encoding='utf-8') as _f:
    _TOOL_SCHEMAS = json.load(_f)

TOOLS_DOCS   = _TOOL_SCHEMAS['docs']
TOOLS_SEARCH = _TOOL_SCHEMAS['search']

# ==========================================
# 2. HELPER FUNCTIONS & JINJA SETUP
# ==========================================

# Ordered agent markers, most specific first. Each entry is
# (agent_id, predicate over the text).
AGENT_MARKERS = [
    (0, lambda t: M_PLANNER in t),
    (2, lambda t: M_SEARCH in t),
    (6, lambda t: M_BROWSER_ACT_A in t and M_BROWSER_ACT_B in t),
    (7, lambda t: M_ANSWERER in t),
    (8, lambda t: M_DOC in t),
]


def match_agent(text, default):
    """First matching marker wins; `default` when nothing matches."""
    for agent_id, hit in AGENT_MARKERS:
        if hit(text):
            return agent_id
    return default


def raise_exception(message):
    raise ValueError(f"Template Error: {message}")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_SCRIPT_DIR),
    trim_blocks=True,
    lstrip_blocks=True
)
env.globals['raise_exception'] = raise_exception

def classify_agent(messages: list) -> int:
    """Agent id from the system prompt: 0 Plan, 2 WebSearch, 6 BrowserAction,
    7 Answerer, 8 Doc, 9 other, -1 no messages."""
    if not messages:
        return -1
    return match_agent(str(messages[0].get('content') or ''), 9)


def get_tools_for_messages(messages):
    """Tool schema for a request, by system prompt. Only these two were ever
    used by gpt-4o requests in this corpus."""
    if not messages or not isinstance(messages, list):
        return None
    system_content = messages[0].get('content', '')
    if M_DOC in system_content:
        return TOOLS_DOCS
    if M_SEARCH in system_content:
        return TOOLS_SEARCH
    return None


def reconstruct_harmony_from_dump(raw_text):
    """Reconstructs Harmony string from ChatCompletionMessage dump."""
    reasoning = None
    reasoning_match = re.search(r"reasoning_content='([\s\S]*?)'\)", raw_text)
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning=['\"]([\s\S]*?)['\"],", raw_text)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).replace('\\n', '\n')

    content = None
    match = re.search(r"content=(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\")", raw_text)
    if match is not None:
        # Determine which group matched (Group 1 for ', Group 2 for ")
        if match.group(1) is not None:
            inner_text = match.group(1)
            quote_used = "'"
        else:
            inner_text = match.group(2)
            quote_used = '"'

        # Reconstruct the string literal exactly as Python sees it
        literal_string = f"{quote_used}{inner_text}{quote_used}"
        content = ast.literal_eval(literal_string)
    tool_calls = []
    tool_pattern = re.compile(r"Function\(arguments='({.*?})',\s*name='([^']+)'\)")
    tool_matches = tool_pattern.findall(raw_text)
    for args, name in tool_matches:
        tool_calls.append({"name": name, "args": args.replace('\\"', '"')})
    harmony_output = []
    if reasoning:
        harmony_output.append(f"<|channel|>analysis<|message|>{reasoning}<|end|>")

    if tool_calls:
        for tool in tool_calls:
            if harmony_output:
                harmony_output.append(f"<|start|>assistant")
            harmony_output.append(
                f"<|channel|>commentary to=functions.{tool['name']} "
                f"<|constrain|>json<|message|>{tool['args']}<|call|>"
            )
    elif content:
        if harmony_output:
            harmony_output.append(f"<|start|>assistant")
        harmony_output.append(
            f"<|channel|>final<|message|>{content}<|return|>"
        )
    return "".join(harmony_output)

# ==========================================
# 3. CORE PROCESSING
# ==========================================

def process_chunk(raw_request_string, raw_output_string, encoding, results):
    try:
        # --- 1. CLEAN THE REQUEST STRING ---
        # The messages list is always on the first line ("Request IN: [...]").
        # Subsequent lines (Config, Elapsed Time, Answer OUT, etc.) also contain
        # brackets, so we must NOT use rfind(']') on the full joined buffer.
        first_line = raw_request_string.split('\n')[0]
        start_idx = first_line.find('[')
        end_idx = first_line.rfind(']')

        if start_idx == -1 or end_idx == -1:
            print("Skipping chunk: Could not find [...] list brackets.")
            return

        clean_list_str = first_line[start_idx : end_idx+1]
        current_messages = ast.literal_eval(clean_list_str)

        # --- 1c. 4o-SPECIFIC: image placeholders + compact tool args ---
        # Sentinel that survives template rendering and is unique enough to
        # never appear in real content.  We split the rendered string at each
        # sentinel and insert IMAGE_PLACEHOLDER_TOKENS zeros inline, so image
        # tokens stay at their exact conversation position (maximises KV-cache
        # prefix reuse across turns).
        image_sentinel_count = 0
        # gpt-4o only: compact tool-call arguments, and swap each inline image
        # for a sentinel so the base64 never reaches the tokenizer.
        for msg in current_messages:
            # Compact tool-call arguments
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool in msg["tool_calls"]:
                    if "function" in tool and "arguments" in tool["function"]:
                        args_dict = json.loads(tool["function"]["arguments"])
                        tool["function"]["arguments"] = json.dumps(
                            args_dict, separators=(',', ':'), ensure_ascii=False
                        )
            # Replace each inline image with the sentinel (preserves position)
            content = msg.get('content')
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if (isinstance(item, dict)
                            and item.get('type') == 'image_url'):
                        url = item.get('image_url', {}).get('url', '')
                        if url.startswith(IMAGE_DATA_URI_PREFIX):
                            new_content.append({'type': 'text', 'text': IMG_SENTINEL})
                            image_sentinel_count += 1
                        else:
                            new_content.append({'type': 'text', 'text': IMAGE_URL_PLACEHOLDER})
                    else:
                        new_content.append(item)
                msg['content'] = new_content

        current_tools = get_tools_for_messages(current_messages)
        agent_type    = classify_agent(current_messages)

        # --- 2. RENDER PREFILL ---
        template = env.get_template(TEMPLATE_4O)
        rendered_prefill = template.render(
            messages=current_messages,
            tools=current_tools,
            builtin_tools=[],
            add_generation_prompt=True
        )
        # Split rendered text at each image sentinel and interleave zeros inline
        if image_sentinel_count > 0:
            parts = rendered_prefill.split(IMG_SENTINEL)
            prefill_ids = []
            for i, part in enumerate(parts):
                prefill_ids += encoding.encode(part, allowed_special="all")
                if i < image_sentinel_count:
                    prefill_ids += [0] * IMAGE_PLACEHOLDER_TOKENS
        else:
            prefill_ids = encoding.encode(rendered_prefill, allowed_special="all")

        # --- 3. PARSE OUTPUT ---
        raw_output_string = raw_output_string.strip()
        decode_str = ""

        if OSS_LIST_MARKER in raw_output_string:
            list_match = re.search(r"(\['\<\|channel\|\>'.*)", raw_output_string)
            if list_match:
                try:
                    token_list = ast.literal_eval(list_match.group(1).strip())
                    decode_str = "".join(token_list)
                except Exception as e:
                    print(f"Error parsing OSS list output: {e}")
                    return
        elif CHAT_MSG_MARKER in raw_output_string:
            decode_str = reconstruct_harmony_from_dump(raw_output_string)

        decode_ids = encoding.encode(decode_str, allowed_special="all")

        # --- 4. COMBINE TOKENS ---
        full_token_ids = prefill_ids + decode_ids

        results.append({
            'num_prefill_tokens': len(prefill_ids),
            'num_decode_tokens': len(decode_ids),
            'agent': agent_type,
            'tokens': str(full_token_ids),
        })

    except Exception as e:
        print(f"Error processing chunk: {e}")
        print(f"  -> First 300 chars of request: {repr(raw_request_string[:300])}")

def compute_deps(results):
    """Fill each row's `dep`: the row indices that must finish before it starts.

    Concurrent OSS rows are sibling children of one parent request: each depends
    on that parent, and the row after them depends on all of them (fan-in).
    Everything else depends on its predecessor; the first row depends on nothing.
    """
    for i, row in enumerate(results):
        if i == 0:
            row['dep'] = '[]'
        elif row['tokens'] == SENTINEL_ASYNC_OSS:
            j = i - 1
            while j >= 0 and results[j]['tokens'] == SENTINEL_ASYNC_OSS:
                j -= 1
            row['dep'] = json.dumps([j] if j >= 0 else [])
        elif results[i - 1]['tokens'] == SENTINEL_ASYNC_OSS:
            j = i - 1
            oss_group = []
            while j >= 0 and results[j]['tokens'] == SENTINEL_ASYNC_OSS:
                oss_group.append(j)
                j -= 1
            row['dep'] = json.dumps(list(reversed(oss_group)))
        else:
            row['dep'] = json.dumps([i - 1])


def verify_csv(output_file):
    """Re-read the CSV and check prefill + decode == len(tokens)"""
    print("\n=== RUNNING VERIFICATION ===")
    verified_count = 0
    error_count = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['tokens'] in (SENTINEL_ASYNC_OSS, SENTINEL_SEQ_OSS):
                verified_count += 1
                continue
            try:
                p_len = int(row['num_prefill_tokens'])
                d_len = int(row['num_decode_tokens'])
                total_len = len(ast.literal_eval(row['tokens']))
                if (p_len + d_len) == total_len:
                    verified_count += 1
                else:
                    print(f"Row {i} MISMATCH: Prefill({p_len}) + Decode({d_len}) != Total({total_len})")
                    error_count += 1
            except Exception as e:
                print(f"Row {i} Error during verification: {e}")
                error_count += 1

    print(f"Verified {verified_count} rows correctly.")
    if error_count == 0:
        print("SUCCESS: All row counts match perfectly.")
    else:
        print(f"FAILURE: {error_count} rows had mismatches.")


def process_file_robust(input_file, output_file):
    print(f"Reading {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    try:
        encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # --- PASS 1: Pre-scan Answer OUT lines → ground truth for row count & order ---
    # (parallel async dispatch can produce multiple Request IN on one line).
    answer_out_line_data = []   # list of (line_idx, '4o'|'oss', decode_count)
    for i, line in enumerate(lines):
        m = ANSWER_OUT_RE.search(line)
        if m:
            model = '4o' if 'gpt-4o' in m.group(2) else 'oss'
            dec   = int(m.group(1)) if m.group(1) else 0
            answer_out_line_data.append((i, model, dec))

    answer_out_seq   = [(model, dec) for _, model, dec in answer_out_line_data]
    ans_line_indices = [li for li, _, _ in answer_out_line_data]  # for dep bisect

    n_oss_expected = sum(1 for model, _ in answer_out_seq if model == 'oss')
    n_4o_expected  = sum(1 for model, _ in answer_out_seq if model == '4o')
    print(f"Answer OUT scan: {len(answer_out_seq)} total  ({n_4o_expected} 4o, {n_oss_expected} OSS)")

    # --- collect sorted list of "Request IN:" line indices (for async detection) ---
    req_in_lines_sorted = sorted(
        i for i, line in enumerate(lines)
        if REQUEST_IN_RE.search(line)
    )

    # --- PASS 2: Chunk parser — collect 4o results and OSS agent types ---
    buffer_request      = []
    buffer_output       = []
    state               = "SEEKING"
    current_is_oss      = False
    fo_results          = []   # 4o rows only

    print("Processing 4o chunks...")

    for line_num, line in enumerate(lines):
        stripped = line.strip()

        req_in_match = REQUEST_IN_RE.search(stripped)
        if req_in_match:
            if state == "READING_OUTPUT" and buffer_request and buffer_output:
                if not current_is_oss:
                    process_chunk("".join(buffer_request), "".join(buffer_output),
                                  encoding, fo_results)

            trimmed_line = stripped[req_in_match.start():] + '\n'
            buffer_request      = [trimmed_line]
            buffer_output       = []
            state               = "READING_REQUEST"
            current_is_oss      = False

        elif CHAT_MSG_MARKER in stripped or stripped.startswith(OSS_LIST_MARKER):
            if state == "READING_REQUEST":
                state = "READING_OUTPUT"
            buffer_output.append(line)

        else:
            if state == "READING_REQUEST":
                if MODEL_4O_ID in line:
                    current_is_oss = False
                elif MODEL_OSS_ID in line:
                    current_is_oss = True
                if not stripped.startswith(COMMENT_PREFIX):
                    buffer_request.append(line)
            elif state == "READING_OUTPUT":
                buffer_output.append(line)

    # Final block
    if buffer_request and buffer_output and not current_is_oss:
        process_chunk("".join(buffer_request), "".join(buffer_output),
                      encoding, fo_results)

    # --- BUILD is_async_oss ---
    # An "OSS group" is a maximal streak of consecutive OSS Answer OUTs, i.e. the
    # OSS calls a single 4o turn made before the next 4o answer; every group is
    # bracketed by 4o on both sides, so it acts as a synchronization block.
    # Within a group, if any adjacent pair has NO "Request IN:" line between them,
    # the whole group was dispatched concurrently (a summarizer fan-out).
    # Sequential calls always have a new Request IN between each Answer OUT.
    oss_seq_indices = [i for i, (model, _) in enumerate(answer_out_seq) if model == 'oss']
    n_oss = len(oss_seq_indices)
    is_async_oss = [False] * n_oss

    if n_oss > 0:
        group_start = 0
        groups = []
        for k in range(1, n_oss):
            if oss_seq_indices[k] != oss_seq_indices[k - 1] + 1:
                groups.append((group_start, k - 1))
                group_start = k
        groups.append((group_start, n_oss - 1))

        for (start, end) in groups:
            if start == end:
                continue  # single isolated OSS → sequential
            group_is_async = False
            for k in range(start, end):
                L_k  = ans_line_indices[oss_seq_indices[k]]
                L_k1 = ans_line_indices[oss_seq_indices[k + 1]]
                idx = bisect.bisect_right(req_in_lines_sorted, L_k)
                has_req_in = idx < len(req_in_lines_sorted) and req_in_lines_sorted[idx] < L_k1
                if not has_req_in:
                    group_is_async = True
                    break
            if group_is_async:
                for k in range(start, end + 1):
                    is_async_oss[k] = True

    # --- BUILD FINAL RESULTS from Answer OUT sequence ---
    # OSS rows: ASYNC_OSS (concurrent) or SEQ_OSS (sequential) based on pre-computation above.
    # 4o  rows: computed token data from chunk parser (in order).
    results = []
    fo_iter = iter(fo_results)
    oss_idx = 0

    for model, dec in answer_out_seq:
        if model == 'oss':
            # OSS agent labels are not decided here: the OWL log does not carry the
            # system prompt. replace_oss_rows.py reads it from the decoded vLLM
            # prompt; rows whose role it does not recognise keep this bucket.
            agent = AGENT_OSS_UNRESOLVED
            tokens = SENTINEL_ASYNC_OSS if is_async_oss[oss_idx] else SENTINEL_SEQ_OSS
            oss_idx += 1
            results.append({'num_prefill_tokens': 0,
                            'num_decode_tokens':  dec,
                            'agent':              agent,
                            'tokens':             tokens})
        else:
            row = next(fo_iter, None)
            if row is not None:
                results.append(row)
            else:
                # 4o chunk parse failed — placeholder with decode count
                results.append({'num_prefill_tokens': 0,
                                'num_decode_tokens':  dec,
                                'agent':              -1,
                                'tokens':             SENTINEL_MISSING_4O})

    compute_deps(results)

    # --- VERIFICATION ---
    actual_oss = sum(1 for r in results if r['tokens'] in (SENTINEL_ASYNC_OSS, SENTINEL_SEQ_OSS))
    actual_4o  = len(results) - actual_oss
    if len(results) != len(answer_out_seq):
        print(f"WARNING: rows {len(results)} != Answer OUT {len(answer_out_seq)}")
    else:
        print(f"Count OK: {len(results)} rows  ({actual_oss} OSS placeholders, {actual_4o} 4o)")

    # --- WRITE CSV ---
    print(f"Writing {len(results)} rows to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    
    print("Done generating CSV.")

    verify_csv(output_file)


def generate_tools_csv(txt_file: Path, tools_file: Path):
    """
    Read a RawText trace and write {stem}_tools.csv listing tool call
    with the index of the session_traces row it feeds into (request_idx).

    observed_elapsed is approximate: the raw log's timing lines carry no tool identity
    and do not appear once per requested tool, so a duration is matched to a call by
    position. Treat it as advisory; merge.py uses the re-measured tool_bench.py times.
    """
    lines = txt_file.read_text(encoding='utf-8', errors='replace').splitlines()

    # Single pass: collect Answer OUT line numbers, Function() claims, and elapsed times.
    ans_out_line_nums = []   # (line_index, 'gpt4o'|'oss') in document order
    # Rows are numbered by Answer OUT order, so the row that issued a tool call is
    # simply the last Answer OUT seen before it.
    claims_api  = []         # non-execute_code: (line_idx, row_idx, name, args_json)
    claims_exec = []         # execute_code:     (line_idx, row_idx, args_json)
    elapsed_api  = []        # (line_idx, float)
    elapsed_code = []        # (line_idx, float)

    for i, line in enumerate(lines):
        m = ANSWER_OUT_RE.search(line)
        if m:
            kind = 'gpt4o' if 'gpt-4o' in m.group(2) else 'oss'
            ans_out_line_nums.append((i, kind))
            continue

        if 'ChatCompletionMessage' in line:
            for args_str, name in FUNCTION_CALL_RE.findall(line):
                if name not in TOOL_NAMES:
                    continue
                try:
                    # Trace files use Python repr escaping: \' inside single-quoted strings.
                    # json.loads rejects \' as an invalid escape, so unescape it.
                    args_json = json.dumps(json.loads(args_str.replace("\\'", "'")), sort_keys=True)
                except Exception:
                    continue
                row_idx = max(0, len(ans_out_line_nums) - 1)
                if name == 'execute_code':
                    claims_exec.append((i, row_idx, args_json))
                else:
                    claims_api.append((i, row_idx, name, args_json))
            continue

        m2 = ELAPSED_TOOL_RE.search(line)
        if m2:
            elapsed_api.append((i, float(m2.group(1))))
            continue

        m3 = ELAPSED_CODE_RE.search(line)
        if m3:
            elapsed_code.append((i, float(m3.group(1))))

    answer_out_seq = [kind for _, kind in ans_out_line_nums]

    # The log has two unlabelled streams: tool-call claims (inside responses) and bare
    # elapsed times. Neither carries an id, so a claim takes the next elapsed after it,
    # skipping ones left by tools we do not track (e.g. return_json_response).
    # The agent stops calling tools once its per-step budget is spent, so elapsed lines stop too.
    def greedy_match(claims, elapsed_list, name_fn):
        rows = []
        e_ptr = 0
        for c_line, row_idx, *c_rest in claims:
            # Advance past elapsed times that appear before this claim (from filtered tools)
            while e_ptr < len(elapsed_list) and elapsed_list[e_ptr][0] <= c_line:
                e_ptr += 1
            if e_ptr >= len(elapsed_list):
                break
            elp_line, elp_val = elapsed_list[e_ptr]
            e_ptr += 1
            rows.append({
                'request_idx':      row_idx,
                'tool_name':        name_fn(c_rest),
                'args_json':        c_rest[-1],
                'observed_elapsed': elp_val,
            })
        return rows

    tool_rows = greedy_match(
        claims_api, elapsed_api,
        name_fn=lambda rest: rest[0]   # rest = (name, args_json)
    )
    exec_rows = greedy_match(
        claims_exec, elapsed_code,
        name_fn=lambda rest: 'execute_code'  # rest = (args_json,)
    )

    # Merge and sort by request_idx so the CSV is ordered by row
    all_rows = sorted(tool_rows + exec_rows, key=lambda r: r['request_idx'])

    with open(tools_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TOOLS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  {tools_file.name}: {len(all_rows)} tool calls across "
          f"{len(answer_out_seq)} requests")


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--tools-only', action='store_true',
                         help='Only generate _tools.csv files; do not overwrite session_traces CSVs')
    _args = _parser.parse_args()

    base    = Path(_SCRIPT_DIR).parent            # GAIATrace/owl
    raw_dir = base.joinpath(*RAW_SUBDIR)
    out_dir = base.joinpath(*OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = base.joinpath(*TOOLS_SUBDIR)
    tools_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(raw_dir.glob(RAW_GLOB)):
        tools_file = tools_dir / TOOLS_CSV_FMT.format(stem=txt_file.stem)

        if _args.tools_only:
            print(f"\n{txt_file.name}")
            generate_tools_csv(txt_file, tools_file)
        else:
            csv_file = out_dir / TRACE_CSV_FMT.format(stem=txt_file.stem)
            print(f"\nProcessing: {txt_file.name} -> {csv_file.name}")
            process_file_robust(str(txt_file), str(csv_file))
            generate_tools_csv(txt_file, tools_file)