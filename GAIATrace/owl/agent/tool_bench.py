"""
tool_bench.py  —  measure how long each tool call actually took

Lives inside the agent tree because it is NOT an independent benchmark: it calls
the OWL toolkits themselves (see make_toolkits), so camel imports resolve from
this directory. Re-measuring needs a .env with the tool API keys, a reachable
vLLM server, and the GAIA attachment files. --summary needs none of that.

Reads  ../raw/tool/tool_calls/L*_tools.csv   (from parser.py)
Writes ../raw/tool/{tool_name}.csv           (append-only, resumable)

Each call is run 3 times, round-robin interleaved to spread rate-limit pressure.
  extract_document_content : effective_elapsed = extraction only, excluding the
                             open-weight summarizer (via _bench_timings)
  ask_question_about_audio : effective_elapsed = whisper-1 only
  all others               : effective_elapsed = full measured time

EXPECTED ERRORS — read this before treating the `error` column as failure.
The summarizer/reasoner LLMs are modelled as part of the serving system, not as
tool cost, so those calls are cut short on purpose. When the cut LLM step raises
(typically ModelProcessingError), the tool call reports an error even though the
part we are timing — extraction, or whisper transcription — already finished and
was recorded in _bench_timings. Such a row is a VALID measurement.

`timing_source` says which it is, so downstream never has to guess:
  split : effective_elapsed came from _bench_timings, i.e. it excludes the LLM.
          Keep it, error or not.
  full  : effective_elapsed is the whole call. With a non-empty error this is a
          fail-fast row carrying no usable timing — exclude it.

The measurements are consumed by ../traces/merge.py, which averages the reps and
fills the inter_request_latency column.

Usage:
  python tool_bench.py [--repeat 3] [--workers 4] [--port 8000] [--gap-secs 1.0]
  python tool_bench.py --summary [--selected-only]     # stats only, no tool calls
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(sys.maxsize)

BASE        = Path(__file__).parent           # GAIATrace/owl/agent
DATA        = BASE.parent                     # GAIATrace/owl
TOOL_DIR    = DATA / "raw" / "tool"           # measurement output
TOOLS_DIR   = TOOL_DIR / "tool_calls"         # {stem}_tools.csv from parser
BENCH_DIR   = TOOL_DIR                        # per-tool measurement CSVs
MANIFEST    = DATA / "traces" / "selected_set.txt"
# camel imports and .env resolve from this directory (we are inside the agent).
REPO_ROOT   = BASE


def _load_dotenv(env_path: Path = REPO_ROOT / '.env'):
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()


def _resolve_path(path: str) -> str:
    if not path or path.startswith('http'):
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    c = REPO_ROOT / p
    return str(c) if c.exists() else path


TARGET_TOOLS = {
    'extract_document_content', 'ask_question_about_image',
    'ask_question_about_audio', 'ask_question_about_video',
    'search_google', 'search_wiki', 'search_wiki_revisions', 'execute_code',
    'search_archived_webpage', 'extract_excel_content',
}

# Identify this client to Wikimedia. The `wikipedia` package sends a generic UA
# ('wikipedia (https://github.com/goldsmith/Wikipedia/)') that Wikimedia's user-agent
# policy rate-limits hard: over the allowance the API answers 403 with a plain-text
# body, which the library json-parses into JSONDecodeError. Measured here: that UA
# fails 20/20 rapid-fire and ~50% at 1s spacing, while a descriptive UA passes 20/20
# — so slowing the benchmark down does not help, only identifying it does.
# The enforcement postdates the traces, so without this a third of search_wiki reps
# record a 0.09s HTTP 403 instead of the ~0.5s call the agent actually made.
# It changes admission, not server work: successful-call latency is unaffected.
WIKI_USER_AGENT = os.environ.get(
    'WIKI_USER_AGENT',
    'GAIATrace-tool-bench/1.0 (https://github.com/psu-paws/GAIATrace; latency measurement)')


def _configure_wikipedia() -> None:
    """Set the Wikimedia UA before any search_wiki / search_wiki_revisions call."""
    try:
        import wikipedia
        wikipedia.set_user_agent(WIKI_USER_AGENT)
    except Exception as e:                     # library absent or API changed
        print(f"warning: could not set wikipedia user-agent ({e})", file=sys.stderr)


# ── Load tool calls from pre-generated _tools.csv files ───────────────────────

def load_tool_calls_from_csv(selected_only: bool = False) -> list[dict]:
    """
    Read all tool_calls/L*_tools.csv files generated by parser.py.
    Returns list of dicts with keys: tool_name, args, observed_elapsed,
    source_file, request_idx, args_key.
    """
    if selected_only:
        if not MANIFEST.exists():
            raise SystemExit("selected_set.txt not found — run select_set.py first")
        stems = set(MANIFEST.read_text(encoding="utf-8").split())
        csv_paths = sorted(p for p in TOOLS_DIR.glob("L*_tools.csv")
                           if p.stem[:-len('_tools')] in stems)
    else:
        csv_paths = sorted(TOOLS_DIR.glob("L*_tools.csv"))

    result = []
    for csv_path in csv_paths:
        source_file = csv_path.stem[:-len('_tools')]  # strip '_tools' suffix
        for row in csv.DictReader(open(csv_path, newline='', encoding='utf-8')):
            tool_name = row['tool_name']
            if tool_name not in TARGET_TOOLS:
                continue
            try:
                args = json.loads(row['args_json'])
            except Exception:
                continue
            result.append({
                'tool_name':        tool_name,
                'args':             args,
                'observed_elapsed': float(row['observed_elapsed']),
                'source_file':      source_file,
                'request_idx':      int(row['request_idx']),
                'args_key':         json.dumps(args, sort_keys=True),
            })
    return result


# ── Arg category ──────────────────────────────────────────────────────────────

def arg_category(tool_name: str, args: dict) -> str:
    from urllib.parse import urlparse as _up
    if tool_name == 'extract_document_content':
        p = args.get('document_path', '')
        if 'youtube.com' in p or 'youtu.be' in p: return 'youtube'
        ext = Path(p).suffix.lower()
        if ext == '.pdf':                        return 'pdf'
        if ext in ('.docx', '.doc'):             return 'docx'
        if ext in ('.xlsx', '.xls', '.csv'):     return 'spreadsheet'
        if ext in ('.pptx', '.ppt'):             return 'pptx'
        if p.startswith('http'):                 return 'web'
        return f'local{ext}' if ext else 'other'
    if tool_name == 'ask_question_about_image':
        p = args.get('image_path', '')
        if p.startswith('http'): return f'url:{_up(p).netloc}'
        return f'local{Path(p).suffix.lower()}' or 'local'
    if tool_name == 'ask_question_about_video':
        p = args.get('video_path', '')
        if 'youtube.com' in p or 'youtu.be' in p: return 'youtube'
        if p.startswith('http'):                   return 'url'
        ext = Path(p).suffix.lower()
        return f'local{ext}' if ext else 'local'
    if tool_name == 'ask_question_about_audio':
        p = args.get('audio_path', '')
        if p.startswith('http'): return 'url'
        return f'local{Path(p).suffix.lower()}' or 'local'
    if tool_name == 'search_google':             return 'query'
    if tool_name in ('search_wiki', 'search_wiki_revisions'): return 'entity'
    if tool_name == 'search_archived_webpage':   return 'archived'
    if tool_name == 'execute_code':              return 'code'
    if tool_name == 'extract_excel_content':
        p = args.get('file_path', args.get('document_path', ''))
        return f'local{Path(p).suffix.lower()}' or 'local'
    return 'unknown'


# ── Toolkit setup ─────────────────────────────────────────────────────────────

def make_toolkits(vllm_url: str) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    _configure_wikipedia()          # see WIKI_USER_AGENT
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from camel.toolkits import (DocumentProcessingToolkit, ImageAnalysisToolkit,
                                AudioAnalysisToolkit, VideoAnalysisToolkit,
                                SearchToolkit, CodeExecutionToolkit)

    gpt4o = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI, model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0})
    gpt_oss = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI, model_type=ModelType.GPT_OSS,
        model_config_dict={"temperature": 0}, url=vllm_url)

    from camel.toolkits import ExcelToolkit
    return {
        'extract_document_content':  DocumentProcessingToolkit(
            cache_dir="tmp", image_analysis_model=gpt4o, text_processing_model=gpt_oss),
        'ask_question_about_image':  ImageAnalysisToolkit(model=gpt4o),
        'ask_question_about_audio':  AudioAnalysisToolkit(cache_dir="tmp/audio",
                                                          audio_reasoning_model=gpt_oss),
        'ask_question_about_video':  VideoAnalysisToolkit(download_directory="tmp/video"),
        'search_google':             SearchToolkit(),
        'search_wiki':               SearchToolkit(),
        'search_wiki_revisions':     SearchToolkit(),
        'search_archived_webpage':   SearchToolkit(),
        'execute_code':              CodeExecutionToolkit(sandbox="subprocess", verbose=True),
        'extract_excel_content':     ExcelToolkit(),
    }


# ── Single run ────────────────────────────────────────────────────────────────

_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

def run_one(tc: dict, toolkits: dict, rep: int) -> dict:
    tool_name = tc['tool_name']
    args      = tc['args']
    toolkit   = toolkits[tool_name]
    cat       = arg_category(tool_name, args)

    def _set_output(r: dict, out) -> None:
        s = str(out)
        r['output_length']  = len(s)
        r['output_preview'] = s[:40].replace('\n', ' ')

    row = {
        'tool_name':         tool_name,
        'arg_category':      cat,
        'source_file':       tc['source_file'],
        'request_idx':       tc['request_idx'],
        'rep':               rep,
        'observed_elapsed':  tc['observed_elapsed'],
        'measured_elapsed':  None,
        'effective_elapsed': None,
        'timing_source':     None,   # 'split' (LLM excluded) or 'full'
        'had_summarization': False,
        'output_length':     None,
        'output_preview':    None,
        'error':             None,
        'args_json':         tc['args_key'],
    }
    t0 = time.perf_counter()
    try:
        if tool_name == 'extract_document_content':
            out = toolkit.extract_document_content(
                document_path=_resolve_path(args['document_path']),
                query=args.get('query', ''))
            content = out[1] if isinstance(out, tuple) else out
            _set_output(row, content)
            row['had_summarization'] = isinstance(content, str) and \
                '(The above is the re-assembled result' in content
            extr = toolkit._bench_timings.get('extraction_elapsed')
            row['effective_elapsed'] = extr if extr is not None else (time.perf_counter() - t0)
            row['timing_source'] = 'split' if extr is not None else 'full'

        elif tool_name == 'ask_question_about_image':
            path = _resolve_path(args.get('image_path', ''))
            if not path.startswith('http') and not Path(path).exists():
                row['error'] = f'local file missing: {path}'
                row['measured_elapsed'] = row['effective_elapsed'] = 0.0
                row['timing_source'] = 'none'
                return row
            out = toolkit.ask_question_about_image(
                image_path=path, question=args.get('question', 'Describe this image.'),
                sys_prompt=args.get('sys_prompt', None))
            _set_output(row, out)

        elif tool_name == 'ask_question_about_audio':
            path = _resolve_path(args.get('audio_path', ''))
            if not path.startswith('http') and not Path(path).exists():
                row['error'] = f'local file missing: {path}'
                row['measured_elapsed'] = row['effective_elapsed'] = 0.0
                row['timing_source'] = 'none'
                return row
            out = toolkit.ask_question_about_audio(
                audio_path=path, question=args.get('question', 'Transcribe this audio.'))
            _set_output(row, out)
            trans = toolkit._bench_timings.get('transcription_elapsed')
            row['effective_elapsed'] = trans if trans is not None else (time.perf_counter() - t0)
            row['timing_source'] = 'split' if trans is not None else 'full'

        elif tool_name == 'ask_question_about_video':
            path = args.get('video_path', '')
            _is_url = path.startswith('http') or 'youtube' in path or 'youtu.be' in path
            if not _is_url and Path(path).suffix.lower() not in _VIDEO_EXTS:
                row['error'] = 'llm_invalid_args'
            out = toolkit.ask_question_about_video(
                video_path=_resolve_path(path),
                question=args.get('question', 'Describe this video.'))
            _set_output(row, out)
            if row['error'] == 'llm_invalid_args':
                row['error'] = None

        elif tool_name == 'search_google':
            out = toolkit.search_google(query=args.get('query', ''))
            _set_output(row, out)

        elif tool_name == 'search_wiki':
            out = toolkit.search_wiki(entity=args.get('entity', ''))
            _set_output(row, out)

        elif tool_name == 'search_wiki_revisions':
            out = toolkit.search_wiki_revisions(**args)
            _set_output(row, out)
        elif tool_name == 'execute_code':
            code_str = args.get('code', args.get('content', ''))
            
            # Unescape literal '\n' characters into real line breaks
            if '\\n' in code_str:
                code_str = code_str.replace('\\n', '\n')
                
            out = toolkit.execute_code(code=code_str)
            _set_output(row, out)
        elif tool_name == 'search_archived_webpage':
            out = toolkit.search_archived_webpage(**args)
            _set_output(row, out)

        elif tool_name == 'extract_excel_content':
            path = _resolve_path(args.get('file_path', args.get('document_path', '')))
            if not Path(path).exists():
                row['error'] = f'local file missing: {path}'
                row['measured_elapsed'] = row['effective_elapsed'] = 0.0
                row['timing_source'] = 'none'
                return row
            out = toolkit.extract_excel_content(document_path=path)
            _set_output(row, out)

        row['measured_elapsed'] = time.perf_counter() - t0
        if row['effective_elapsed'] is None:
            row['effective_elapsed'] = row['measured_elapsed']
            row['timing_source'] = 'full'

    except Exception as e:
        row['measured_elapsed'] = time.perf_counter() - t0
        # Recover split latency from _bench_timings even when the full call failed
        # (e.g. extraction succeeded but OSS summarizer/reasoner refused connection)
        if row['effective_elapsed'] is None:
            timings = getattr(toolkit, '_bench_timings', {})
            split = timings.get('extraction_elapsed') or timings.get('transcription_elapsed')
            row['effective_elapsed'] = split if split is not None else row['measured_elapsed']
            # 'split' here = the timed part finished and only the cut LLM step failed.
            row['timing_source'] = 'split' if split is not None else 'full'
        if row.get('error') != 'llm_invalid_args':
            row['error'] = f'{type(e).__name__}: {e}'

    return row


# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDNAMES = ['tool_name', 'arg_category', 'source_file', 'request_idx', 'rep',
              'observed_elapsed', 'measured_elapsed', 'effective_elapsed',
              'timing_source', 'had_summarization', 'output_length', 'output_preview',
              'error', 'args_json']


def open_writer(out_dir: Path, tool_name: str):
    path = out_dir / f"{tool_name}.csv"
    exists = path.exists()
    fh = open(path, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction='ignore')
    if not exists:
        w.writeheader()
    return fh, w


def done_counts(out_dir: Path) -> dict:
    done = defaultdict(int)
    for tool_name in TARGET_TOOLS:
        path = out_dir / f"{tool_name}.csv"
        if not path.exists():
            continue
        for row in csv.DictReader(open(path, newline='', encoding='utf-8')):
            done[(row['tool_name'], row.get('source_file', ''), row.get('request_idx', ''), row['args_json'])] += 1
    return done


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summaries(out_dir: Path) -> list:
    """Write summary_full.txt (every run measured) and, when selected_set.txt is
    present, summary_set.txt (only the runs the dataset actually uses)."""
    written = []
    full = out_dir / 'summary_full.txt'
    full.write_text(build_summary(out_dir), encoding='utf-8')
    written.append(full)
    if MANIFEST.exists():
        stems = set(MANIFEST.read_text(encoding='utf-8').split())
        sub = out_dir / 'summary_set.txt'
        sub.write_text(build_summary(out_dir, selected_stems=stems), encoding='utf-8')
        written.append(sub)
    return written


def build_summary(out_dir: Path, selected_stems: set = None) -> str:
    eff_by = defaultdict(list); err_by = defaultdict(int)
    inv_by = defaultdict(int)

    for csv_path in sorted(out_dir.glob('*.csv')):
        for row in csv.DictReader(open(csv_path, newline='', encoding='utf-8')):
            if selected_stems is not None:
                stem = row.get('source_file', '').replace('.csv', '')
                if stem not in selected_stems:
                    continue
            key = (row['tool_name'], row.get('arg_category', '?'))
            err = row.get('error', '')
            if err == 'llm_invalid_args':
                inv_by[key] += 1; continue
            # An error with timing_source == 'split' is the intended early stop:
            # the timed part finished and only the cut LLM step raised. Keep it.
            # An error with a full-call time is a fail-fast row with no usable timing.
            if err and row.get('timing_source') != 'split':
                err_by[key] += 1
                continue
            try:
                eff_str = row.get('effective_elapsed') or row.get('measured_elapsed')
                eff_val = float(eff_str) if eff_str else None
                if eff_val is not None and eff_val > 0:
                    eff_by[key].append(eff_val)
                elif err:
                    err_by[key] += 1
            except (ValueError, TypeError):
                if err:
                    err_by[key] += 1

    all_keys = sorted(set(list(eff_by) + list(err_by) + list(inv_by)))
    if not all_keys:
        return "No data.\n"

    hdr = (f"{'Tool':<30} {'Category':<22} {'N':>5} {'Err':>5}"
           f"  {'Eff min':>8} {'Eff max':>8} {'Eff μ±σ':<18}")
    lines = [hdr, "-" * len(hdr)]

    def _fmt_row(tool, cat, eff, err):
        n = len(eff)
        if n == 0:
            return f"{tool:<30} {cat:<22} {n:>5} {err:>5}   (all errors)"
        em = statistics.mean(eff)
        es = statistics.stdev(eff) if n > 1 else 0.0
        emin = min(eff); emax = max(eff)
        return (f"{tool:<30} {cat:<22} {n:>5} {err:>5}"
                f"  {emin:>8.3f} {emax:>8.3f} {em:>8.3f}±{es:<8.3f}")

    # Group keys by tool name
    tools_seen = []
    by_tool: dict[str, list] = defaultdict(list)
    for key in all_keys:
        tool = key[0]
        if tool not in by_tool:
            tools_seen.append(tool)
        by_tool[tool].append(key)

    for tool in tools_seen:
        keys = by_tool[tool]
        t_eff, t_err = [], 0
        for key in keys:
            _, cat = key
            n = len(eff_by[key]); err = err_by[key]; inv = inv_by[key]
            if inv and n == 0 and err == 0:
                lines.append(f"{tool:<30} {cat:<22} {n:>5} {inv:>5}   (excluded: llm_invalid_args)")
                continue
            t_err += err
            t_eff += eff_by[key]
            lines.append(_fmt_row(tool, cat, eff_by[key], err))
        # Tool-level total row
        lines.append(_fmt_row(f"  {tool} [TOTAL]", "", t_eff, t_err))
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Rate limiting ─────────────────────────────────────────────────────────────

def make_throttle(base_gap: float):
    _last: dict[str, float] = {}
    _lock = threading.Lock()
    _gaps = {'firecrawl': base_gap * 2.0, 'gemini': base_gap * 2.0,
             'openai': base_gap, 'search': base_gap * 0.5, 'local': 1.5}

    def _svc(tool_name, args):
        if tool_name in ('search_google', 'search_wiki', 'search_wiki_revisions',
                         'search_archived_webpage'):
            return 'search'
        if tool_name == 'extract_document_content':
            p = args.get('document_path', '')
            return 'firecrawl' if (p.startswith('http') or 'youtube' in p) else 'local'
        if tool_name in ('ask_question_about_image', 'ask_question_about_audio'):
            return 'openai'
        if tool_name == 'ask_question_about_video':
            return 'gemini'
        return 'local'

    def throttle(tool_name, args):
        svc = _svc(tool_name, args)
        gap = _gaps.get(svc, 0.0)
        if gap <= 0:
            return
        with _lock:
            wait = gap - (time.time() - _last.get(svc, 0.0))
            if wait > 0:
                time.sleep(wait)
            _last[svc] = time.time()

    return throttle


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat',    type=int,   default=3)
    ap.add_argument('--workers',   type=int,   default=4)
    ap.add_argument('--port',      type=int,   default=8000)
    ap.add_argument('--vllm-host', default='10.136.74.50')
    ap.add_argument('--gap-secs',  type=float, default=2.0,
                    help='Base gap between calls to same service (default 1s; '
                         'firecrawl/gemini get 2x, search gets 0.5x)')
    ap.add_argument('--out-dir',   default=str(BENCH_DIR))
    ap.add_argument('--tools',     nargs='+', metavar='TOOL',
                    help='Only benchmark these tool names (default: all)')
    ap.add_argument('--summary',       action='store_true',
                    help='Print summary from existing CSVs and exit')
    ap.add_argument('--selected-only', action='store_true',
                    help='Only benchmark tool calls from selected_set.txt stems')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        sel = (set(MANIFEST.read_text(encoding="utf-8").split())
               if args.selected_only and MANIFEST.exists() else None)
        print(build_summary(out_dir, selected_stems=sel))
        # stderr: run_all.sh redirects stdout into table2_owl.txt, and these are
        # progress lines, not part of the table.
        for p in write_summaries(out_dir):
            print(f"Written: {p}", file=sys.stderr)
        return

    print("Loading tool calls from _tools.csv files …")
    all_calls = load_tool_calls_from_csv(selected_only=args.selected_only)
    if args.tools:
        filter_set = set(args.tools)
        all_calls = [tc for tc in all_calls if tc['tool_name'] in filter_set]
        print(f"Filtered to {len(all_calls)} calls for tools: {sorted(filter_set)}")

    by_tool = defaultdict(int)
    for tc in all_calls:
        by_tool[tc['tool_name']] += 1
    print(f"Total tool calls: {len(all_calls)}")
    for t, n in sorted(by_tool.items()):
        print(f"  {t}: {n}")

    done = done_counts(out_dir)
    total_done = sum(done.values())
    if total_done:
        print(f"\nResuming — {total_done} reps already completed.")

    # Build interleaved work list: rep-1 of everything, then rep-2, …
    rounds: list[list] = [[] for _ in range(args.repeat)]
    for tc in all_calls:
        completed = done.get((tc['tool_name'], tc['source_file'], str(tc['request_idx']), tc['args_key']), 0)
        for rep in range(1, args.repeat + 1):
            if rep > completed:
                rounds[rep - 1].append((tc, rep))

    work_items = [item for r in rounds for item in r]
    if not work_items:
        print("All runs complete.")
        print(build_summary(out_dir))
        return

    print(f"\n{len(work_items)} runs to execute "
          f"({args.repeat} reps × {len(all_calls)} calls)\n")

    vllm_url = f"http://{args.vllm_host}:{args.port}/v1"
    throttle = make_throttle(args.gap_secs)

    # Each thread gets its own toolkit instances to avoid _bench_timings races.
    _thread_local = threading.local()

    def get_toolkits():
        if not hasattr(_thread_local, 'toolkits'):
            _thread_local.toolkits = make_toolkits(vllm_url)
        return _thread_local.toolkits

    fhs, writers = {}, {}
    for t in TARGET_TOOLS:
        fhs[t], writers[t] = open_writer(out_dir, t)

    lock = threading.Lock()
    completed = 0
    total = len(work_items)

    def do_work(item):
        tc, rep = item
        throttle(tc['tool_name'], tc['args'])
        return run_one(tc, get_toolkits(), rep)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_work, item): item for item in work_items}
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                writers[row['tool_name']].writerow(row)
                fhs[row['tool_name']].flush()
                completed += 1
                if completed % 50 == 0 or completed == total:
                    flag = ' [ERR]' if row.get('error') else ''
                    eff = row.get('effective_elapsed') or 0
                    print(f"[{completed}/{total}] {row['tool_name']} "
                          f"{row['arg_category']} rep={row['rep']} "
                          f"eff={eff:.2f}s{flag}")

    for fh in fhs.values():
        fh.close()

    print(f"\n{build_summary(out_dir)}")
    for p in write_summaries(out_dir):
        print(f"Written: {p}")
    print(f"Done. Results in {out_dir}/")


if __name__ == '__main__':
    main()
