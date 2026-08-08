r"""
replace_oss_rows.py — fill in the OSS rows parser.py left as placeholders.

The OWL log records only a decode count for open-weight calls, so parser.py emits
ASYNC_OSS/SEQ_OSS placeholders. The real token ids come from the vLLM server log
(raw/vllm/), which is the ground truth for those requests.

Per CSV in traces/session_traces/:
  - read the OSS decode sequence from raw/agent/<stem>.txt;
  - take raw/vllm/<stem>.txt, the vLLM log for that session (same stem);
  - pair each placeholder with the closest unused vLLM row by decode count, within
    DECODE_TOLERANCE — any slot left unmatched skips the file rather than guessing;
  - write back prefill, decode and tokens, plus `agent` when the decoded prompt
    identifies the role.
"""

import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# ==========================================
# 0. CONFIGURATION
# Everything corpus-, model- or format-specific lives here.
# Change these rather than the code body below.
# ==========================================

# --- Paths ------------------------------------------------------------------
PARSED_SUBDIR = ("raw", "vllm")               # parsed OSS sessions, named by CSV stem
CSV_SUBDIR    = ("traces", "session_traces")  # parser output, edited in place
TRACE_SUBDIR  = ("raw", "agent")              # OWL stdout logs

BASE        = Path(__file__).parent.parent    # GAIATrace/owl
PARSED_ROOT = BASE.joinpath(*PARSED_SUBDIR)
CSV_DIR     = BASE.joinpath(*CSV_SUBDIR)
TRACE_DIR   = BASE.joinpath(*TRACE_SUBDIR)

# --- Markers ----------------------------------------------------------------
# Decode-token counts from the OWL log; the sequence of these is what identifies
# which parsed session belongs to which CSV.
TRACE_DECODE_RE  = re.compile(r'Answer OUT\d+:\s+(\d+)\s+openai/gpt-oss-120b')
# Row format written by parser's format_pair(): "n_pre,n_dec,\"[ids]\""
PARSED_ROW_RE    = re.compile(r'^(\d+),(\d+),"?(\[.*)')
# Harmony prompt start tokens — identifies an already-filled OSS row.
OSS_TOKEN_PREFIX = "[200006, 17360, 200008,"
# Placeholders written by parser for rows awaiting real token ids.
OSS_PLACEHOLDERS = ("ASYNC_OSS", "SEQ_OSS")

# --- CSV columns ------------------------------------------------------------
COL_PREFILL  = "num_prefill_tokens"
COL_DECODE   = "num_decode_tokens"
COL_TOKENS   = "tokens"
COL_AGENT    = "agent"
DROP_COLUMNS = {"cached_prefill_tokens", "unseen_prefill_tokens"}
DEFAULT_FIELDNAMES = ["num_prefill_tokens", "num_decode_tokens", "agent", "dep", "tokens"]

# --- Matching tolerances ----------------------------------------------------
DECODE_TOLERANCE  = 0.05   # max relative decode-count difference for a match
PREFILL_WEIGHT    = 0.1    # prefill contribution to the tie-break score


# ---------------------------------------------------------------------------
# Parsed session index
# ---------------------------------------------------------------------------

# The 4o-only roles (planner, browser, answerer, document) are matched by 
# parser.py's AGENT_MARKERS and never appear here.
_AGENT_TEXT_MARKERS = [
    (4, 'I have retrieved some information from a long document'),
    (1, 'You are coordinating a group of workers'),
    (3, 'specializes in reasoning'),
    (2, 'search the web'),
    (5, 'planning complex tasks'),
]


class ParsedSession:
    def __init__(self, path: Path, rows: list[tuple], texts: list[str]):
        self.path = path
        self.rows  = rows    # list of (n_pre, n_dec, tokens_str)
        self.texts = texts   # decoded text for each row (same length)


def load_parsed_sessions() -> list[ParsedSession]:
    sessions = []
    for pf in sorted(PARSED_ROOT.rglob("*.txt")):
        rows  = []
        texts = []
        pending_token_row = False
        with open(pf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = PARSED_ROW_RE.match(line.rstrip())
                if m:
                    tokens = m.group(3).rstrip('"')
                    rows.append((int(m.group(1)), int(m.group(2)), tokens))
                    pending_token_row = True
                elif pending_token_row:
                    texts.append(line.strip())
                    pending_token_row = False
        # Pad texts if any token lines had no following text line
        while len(texts) < len(rows):
            texts.append('')
        if rows:
            sessions.append(ParsedSession(pf, rows, texts))
    return sessions


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def is_oss_row(row: dict) -> bool:
    t = row.get(COL_TOKENS, "")
    return t in OSS_PLACEHOLDERS or t.startswith(OSS_TOKEN_PREFIX)



# ---------------------------------------------------------------------------
# Replace one CSV
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, session: ParsedSession) -> str:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    oss_indices = [i for i, r in enumerate(rows) if is_oss_row(r)]
    parsed_rows = session.rows

    def _apply_parsed_row(target: dict, n_pre: int, n_dec: int,
                          tokens: str, text: str):
        target[COL_PREFILL] = n_pre
        target[COL_DECODE]  = n_dec
        target[COL_TOKENS]  = tokens
        detected = _agent_from_text(text)
        if detected is not None:
            target[COL_AGENT] = detected

    if len(parsed_rows) < len(oss_indices):
        print(f"  WARNING: {csv_path.name}: parsed has FEWER rows than CSV OSS slots "
              f"({len(parsed_rows)} parsed < {len(oss_indices)} OSS) — "
              f"cannot fill all placeholders, skipping!")
        return (f"SKIP parsed {len(parsed_rows)} < CSV OSS {len(oss_indices)}")

    # Match each CSV OSS position to the closest parsed row by (prefill, decode) counts.
    # Greedily pick the best remaining parsed row for each CSV OSS slot in order.
    # Accepts matches within 5% on decode; unmatched slots cause a skip.
    TOL = DECODE_TOLERANCE
    remaining = list(enumerate(parsed_rows))  # (orig_idx, (n_pre, n_dec, tokens))
    matched = []
    for csv_idx in oss_indices:
        exp_pre = int(rows[csv_idx].get(COL_PREFILL) or 0)
        exp_dec = int(rows[csv_idx].get(COL_DECODE) or 0)
        best_j, best_score = -1, float("inf")
        for j, (orig_i, (n_pre, n_dec, _tokens)) in enumerate(remaining):
            dec_diff = abs(n_dec - exp_dec) / max(exp_dec, 1)
            pre_diff = abs(n_pre - exp_pre) / max(exp_pre, 1) if exp_pre else 0.0
            score = dec_diff + pre_diff * PREFILL_WEIGHT  # decode is primary
            if dec_diff <= TOL and score < best_score:
                best_score, best_j = score, j
        if best_j == -1:
            return (f"SKIP no close match for OSS slot dec={exp_dec} "
                    f"among {len(remaining)} remaining parsed rows")
        orig_i, (n_pre, n_dec, tokens) = remaining.pop(best_j)
        matched.append((orig_i, n_pre, n_dec, tokens))

    for i, csv_idx in enumerate(oss_indices):
        orig_i, n_pre, n_dec, tokens = matched[i]
        text = session.texts[orig_i] if orig_i < len(session.texts) else ''
        _apply_parsed_row(rows[csv_idx], n_pre, n_dec, tokens, text)
    _write_csv(csv_path, rows)
    extra = len(parsed_rows) - len(oss_indices)
    suffix = f" ({extra} parsed skipped)" if extra else ""
    return f"OK {len(oss_indices)} rows replaced{suffix}"


def _agent_from_text(text: str) -> int | None:
    """
    Return agent type detected from the decoded session text, or None.
    Checks the marker list in priority order; only overwrites the CSV's
    existing agent value when a match is found.
    """
    for agent_id, marker in _AGENT_TEXT_MARKERS:
        if marker in text:
            return agent_id
    return None


def _write_csv(csv_path: Path, rows: list[dict]):
    fieldnames = [k for k in (rows[0].keys() if rows else DEFAULT_FIELDNAMES)
                  if k not in DROP_COLUMNS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading parsed sessions …")
    sessions = load_parsed_sessions()
    # Files in raw/vllm are named after the CSV stem they belong to — that naming
    # was established once by matching decode-count sequences, so lookup is direct.
    sessions_by_stem = {s.path.stem: s for s in sessions}
    print(f"  {len(sessions)} sessions loaded.\n")

    ok = skipped = 0

    for csv_path in sorted(CSV_DIR.glob("*.csv")):
        stem  = csv_path.stem
        trace_path = TRACE_DIR / f"{stem}.txt"
        if not trace_path.exists():
            print(f"  {csv_path.name}: trace file missing")
            skipped += 1
            continue

        with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
            has_oss = any(TRACE_DECODE_RE.search(line) for line in f)

        if not has_oss:
            # Pure-4o file — no OSS rows to fill, but still normalise the columns
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                _write_csv(csv_path, rows)
                print(f"  {csv_path.name}: no OSS calls — columns normalised")
            continue

        session = sessions_by_stem.get(stem)
        if session is None:
            print(f"  {csv_path.name}: no vLLM log named {stem}.txt")
            skipped += 1
            continue

        status = process_csv(csv_path, session)
        print(f"  {csv_path.name}: {status}  [{session.path.parent.name}/{session.path.name}]")
        if status.startswith("OK"):
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} updated, {skipped} skipped.")
