"""
accuracy.py

Success rate over the selected set, and how token cost splits between
successful and failed runs.

The rate here is "solved by at least one attempt", not pass@1. Collection retried
a question until one run solved it, then moved on, so failures accumulate extra
attempts while successes stop early; select_set.py then keeps the successful run
where one exists. That is deliberate. The dataset is meant to be a corpus of
usable traces, not a benchmark score.

Input:
    traces/selected_set.txt          the set (one run per question)
    traces/session_traces/*.csv      per-request tokens
    traces/run_outcomes.csv          solved/not verdict per run
Usage:
    python accuracy.py
Output:
    outputs/accuracy.txt
"""

import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

BASE        = Path(__file__).parent          # GAIATrace/owl/analysis
DATA        = BASE.parent                    # GAIATrace/owl
TRACES_DIR  = DATA / "traces" / "session_traces"
OUTCOMES    = DATA / "traces" / "run_outcomes.csv"   # per-run solved/not verdict
MANIFEST    = DATA / "traces" / "selected_set.txt"
OUT_DIR     = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE    = OUT_DIR / "accuracy.txt"

MODEL_NAME = {0: "gpt-oss-120b", 1: "gpt-4o"}


_OUTCOMES = None

def get_score(stem):
    """True/False if the run was solved, None if it has no recorded verdict.

    Read from traces/run_outcomes.csv rather than the agent logs: the logs are
    plain-text GAIA content and are not shipped with the artifact (see README).
    The verdict is the only thing this ever needed from them.
    """
    global _OUTCOMES
    if _OUTCOMES is None:
        with open(OUTCOMES, encoding="utf-8") as f:
            _OUTCOMES = {r["source_file"]: r["solved"] == "True"
                         for r in csv.DictReader(f)}
    return _OUTCOMES.get(stem)


def model_id(tok):
    if tok.startswith("[200006, 77944"):
        return 1
    if tok.startswith("[200006, 17360"):
        return 0
    return -1


def tokens_of(stem):
    """Per-model (requests, prefill, decode) for one trace."""
    acc = {0: [0, 0, 0], 1: [0, 0, 0]}
    with open(TRACES_DIR / f"{stem}.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = int(row["num_prefill_tokens"])
                d = int(row["num_decode_tokens"])
            except (ValueError, KeyError):
                continue
            mid = model_id(row.get("tokens", ""))
            if mid == -1:
                continue
            acc[mid][0] += 1
            acc[mid][1] += p
            acc[mid][2] += d
    return acc


# ── Collect ───────────────────────────────────────────────────────────────────

if not MANIFEST.exists():
    raise SystemExit("selected_set.txt not found — run select_set.py first")
stems = [s for s in MANIFEST.read_text(encoding="utf-8").split() if s]

groups = {True: [], False: [], None: []}
per_stem = {}
for stem in stems:
    if not (TRACES_DIR / f"{stem}.csv").exists():
        continue
    sc = get_score(stem)
    groups[sc].append(stem)
    per_stem[stem] = tokens_of(stem)

n_total   = sum(len(v) for v in groups.values())
n_success = len(groups[True])


def attempt_stats():
    """Raw success rate over every run collected, not just the selected set."""
    stems = [p.stem for p in sorted(TRACES_DIR.glob("*.csv"))]
    return len(stems), sum(1 for s in stems if get_score(s))


def totals(stem_list):
    """(requests, prefill, decode) summed over a group, plus per-model."""
    overall = [0, 0, 0]
    by_model = {0: [0, 0, 0], 1: [0, 0, 0]}
    for stem in stem_list:
        for mid, (r, p, d) in per_stem[stem].items():
            by_model[mid][0] += r
            by_model[mid][1] += p
            by_model[mid][2] += d
            overall[0] += r
            overall[1] += p
            overall[2] += d
    return overall, by_model


all_overall, all_by_model = totals(list(per_stem))
grand_prefill = all_overall[1]
grand_decode  = all_overall[2]
grand_tokens  = grand_prefill + grand_decode


def pct(part, whole):
    return part / whole * 100 if whole else 0.0

# ── Report ────────────────────────────────────────────────────────────────────

L = []
L.append("=" * 90)
L.append("  GAIA SUCCESS RATE AND TOKEN COST  (selected set, best run per question)")
L.append("=" * 90)
L.append("")
L.append(f"  Sessions       : {n_total}")
L.append(f"  Success        : {n_success}")
L.append(f"  Failed         : {len(groups[False])}")
if groups[None]:
    L.append(f"  No score found : {len(groups[None])}")
L.append(f"  Selected       : {n_success}/{n_total} = {n_success / n_total * 100:.1f}%")

_runs, _ok = attempt_stats()
L.append(f"  All runs       : {_ok}/{_runs} = {_ok / _runs * 100:.1f}%")
L.append("")
L.append(f"  Total tokens across the set: {grand_tokens:,}"
         f"  (prefill {all_overall[1]:,} + decode {all_overall[2]:,})")

L.append("")
L.append("-" * 90)
L.append("  TOKENS BY OUTCOME")
L.append("-" * 90)
L.append(f"  {'Outcome':<9} {'Runs':>5} {'Reqs':>6} {'Prefill':>13} {'Pre%':>6} "
         f"{'Decode':>11} {'Dec%':>6} {'Total':>13} {'Tot%':>6}")

for label, key in (("Success", True), ("Failed", False), ("No score", None)):
    if not groups[key]:
        continue
    ov, _ = totals(groups[key])
    tot = ov[1] + ov[2]
    L.append(f"  {label:<9} {len(groups[key]):>5} {ov[0]:>6} "
             f"{ov[1]:>13,} {pct(ov[1], grand_prefill):>5.1f}% "
             f"{ov[2]:>11,} {pct(ov[2], grand_decode):>5.1f}% "
             f"{tot:>13,} {pct(tot, grand_tokens):>5.1f}%")

L.append("")
L.append("  Per run (mean tokens):")
for label, key in (("Success", True), ("Failed", False)):
    if not groups[key]:
        continue
    ov, _ = totals(groups[key])
    n = len(groups[key])
    L.append(f"    {label:<8} prefill {ov[1]/n:>12,.0f}   decode {ov[2]/n:>9,.0f}   "
             f"total {(ov[1]+ov[2])/n:>12,.0f}   reqs {ov[0]/n:>5.1f}")

L.append("")
L.append("-" * 90)
L.append("  TOKENS BY OUTCOME AND MODEL")
L.append("-" * 90)
L.append(f"  {'Outcome':<9} {'Model':<13} {'Reqs':>6} {'Prefill':>13} {'Pre%':>6} "
         f"{'Decode':>11} {'Dec%':>6} {'Tot%':>6}")
for label, key in (("Success", True), ("Failed", False)):
    if not groups[key]:
        continue
    _, bm = totals(groups[key])
    for mid in (0, 1):
        r, p, d = bm[mid]
        if r == 0:
            continue
        L.append(f"  {label:<9} {MODEL_NAME[mid]:<13} {r:>6} "
                 f"{p:>13,} {pct(p, grand_prefill):>5.1f}% "
                 f"{d:>11,} {pct(d, grand_decode):>5.1f}% "
                 f"{pct(p + d, grand_tokens):>5.1f}%")

L.append("")
L.append("=" * 90)

text = "\n".join(L)
print(text)
OUT_FILE.write_text(text + "\n", encoding="utf-8")
print(f"\nWritten: {OUT_FILE}")
