"""
score_gaia.py

GAIA accuracy over the MiroThinker set, and how token cost splits between
correct and incorrect runs.

The GAIA ground truth is not redistributed here. Fetch the 2023 validation set
from https://huggingface.co/datasets/gaia-benchmark/GAIA and point --ground-truth
at a JSONL of {"task_id": ..., "ground_truth": ...} records, one per line; the
103 text-only tasks are the ones this set covers.

Input:
    --ground-truth PATH              default data/gaia_text103.jsonl (not shipped)
    mirothinker/traces/transcripts/*.txt     transcripts, one per task
    mirothinker/traces/session_traces/*.csv   per-request tokens
Usage:
    python score_gaia.py --ground-truth /path/to/gaia_text103.jsonl
Output:
    outputs/score_gaia.txt           per-task verdicts, accuracy, tokens by outcome
"""

import argparse
import csv
import json
import re
import string
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

BASE        = Path(__file__).resolve().parent      # GAIATrace/mirothinker/analysis
DATA        = BASE.parent                          # GAIATrace/mirothinker
ROOT        = DATA.parent                          # GAIATrace
RAWTEXT_DIR = DATA / "traces" / "transcripts"
TRACES_DIR  = DATA / "traces" / "session_traces"
GROUND_TRUTH = ROOT / "data" / "gaia_text103.jsonl"
OUT_DIR     = BASE / "outputs"

def norm_num(s):
    for c in ["$", "%", ","]:
        s = s.replace(c, "")
    try:
        return float(s)
    except ValueError:
        return None

def norm_str(s, rm_punct=True):
    s = s.lower().strip()
    if rm_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    return s.strip()

def is_float(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def gaia_score(pred, truth):
    if is_float(truth):
        n = norm_num(str(pred))
        return n is not None and n == float(truth)
    if any(c in truth for c in [",", ";"]):
        gt_e = [e.strip() for e in re.split(r"[,;]", truth)]
        ma_e = [e.strip() for e in re.split(r"[,;]", str(pred))]
        if len(gt_e) != len(ma_e):
            return False
        for g, m in zip(gt_e, ma_e):
            if is_float(g):
                n = norm_num(m)
                if n is None or n != float(g):
                    return False
            else:
                if norm_str(m, rm_punct=False) != norm_str(g, rm_punct=False):
                    return False
        return True
    return norm_str(str(pred)) == norm_str(truth)

BOXED_MARKER = "\\boxed{"

def extract_boxed(path):
    last = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            i = 0
            while True:
                i = line.find(BOXED_MARKER, i)
                if i == -1:
                    break
                start = i + len(BOXED_MARKER)
                depth, j = 1, start
                while j < len(line) and depth:
                    if line[j] == "{":
                        depth += 1
                    elif line[j] == "}":
                        depth -= 1
                    j += 1
                if depth == 0:                     # unbalanced = line truncated
                    last = line[start:j - 1].strip()
                i = start
    return last

def count_tokens(traces_dir, task_ids):
    """(prefill, decode, missing) summed over the given tasks."""
    prefill = decode = 0
    missing = []
    for tid in task_ids:
        path = traces_dir / f"{tid}.csv"
        if not path.exists():
            missing.append(tid)
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prefill += int(row.get("num_prefill_tokens") or 0)
                decode  += int(row.get("num_decode_tokens")  or 0)
    return prefill, decode, missing

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--raw",    type=Path, default=RAWTEXT_DIR,  help="dir of <task_id>.txt transcripts")
    ap.add_argument("--traces", type=Path, default=TRACES_DIR,   help="dir of <task_id>.csv traces")
    ap.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH,
                    help="JSONL of {task_id, ground_truth}; not shipped, see module docstring")
    args = ap.parse_args()

    for p in (args.raw, args.traces):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    if not args.ground_truth.exists():
        raise SystemExit(
            f"ground truth not found: {args.ground_truth}\n"
            "pass --ground-truth /path/to/file.jsonl ")

    gt = {}
    with open(args.ground_truth, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt[rec["task_id"]] = rec["ground_truth"]

    correct_ids, incorrect_ids, unscored = [], [], []
    rows = []
    for path in sorted(args.raw.glob("*.txt")):
        tid = path.stem
        if tid not in gt:
            unscored.append(tid)
            continue
        pred = extract_boxed(path)
        ok = gaia_score("" if pred is None else str(pred), gt[tid])
        (correct_ids if ok else incorrect_ids).append(tid)
        rows.append((tid, pred, gt[tid], ok))

    total = len(rows)
    if total == 0:
        raise SystemExit(f"no transcripts in {args.raw} matched {args.ground_truth.name}")

    L = []
    L.append("=" * 110)
    L.append("  GAIA ACCURACY AND TOKEN COST  (MiroThinker, one run per question)")
    L.append("=" * 110)
    L.append("")
    L.append(f"  {'task_id':36s} {'':1s}  {'predicted':45s} {'ground truth'}")
    L.append("-" * 110)
    for tid, pred, truth, ok in rows:
        L.append(f"  {tid}  {'o' if ok else 'x'}  {str(pred)[:44]:45s} {truth}")
    L.append("-" * 110)
    L.append("")
    L.append(f"  Tasks in ground truth : {len(gt)}")
    L.append(f"  Scored                : {total}")
    if len(gt) != total:
        L.append(f"  Missing transcripts   : {len(gt) - total}")
    if unscored:
        L.append(f"  Transcripts not in GT : {len(unscored)}")
    L.append(f"  Accuracy              : {len(correct_ids)}/{total} = {len(correct_ids)/total*100:.1f}%")

    tp, td, miss_t = count_tokens(args.traces, correct_ids)
    fp, fd, miss_f = count_tokens(args.traces, incorrect_ids)
    grand = tp + td + fp + fd
    if grand == 0:
        raise SystemExit(f"no token rows found under {args.traces} — wrong --traces dir?")

    L.append("")
    L.append("-" * 110)
    L.append("  TOKENS BY OUTCOME")
    L.append("-" * 110)
    L.append(f"  {'Outcome':<20} {'Prefill':>15} {'Decode':>15} {'Total':>15} {'Tot%':>7}")
    for label, n, p, d in (("Correct", len(correct_ids), tp, td),
                           ("Incorrect", len(incorrect_ids), fp, fd)):
        L.append(f"  {label + f' ({n} tasks)':<20} {p:>15,} {d:>15,} {p+d:>15,} "
                 f"{(p+d)/grand*100:>6.1f}%")
    L.append(f"  {'Total':<20} {tp+fp:>15,} {td+fd:>15,} {grand:>15,} {100.0:>6.1f}%")

    missing = miss_t + miss_f
    if missing:
        L.append("")
        L.append(f"  WARNING: {len(missing)} scored tasks have no trace CSV and "
                 f"contribute 0 tokens: {', '.join(missing[:5])}"
                 + (" ..." if len(missing) > 5 else ""))

    L.append("")
    L.append("=" * 110)
    text = "\n".join(L)
    print(text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "score_gaia.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
