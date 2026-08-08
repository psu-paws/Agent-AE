"""
select_set.py — choose one run per task
Single source of truth for "set" selection (one best run per question).

Selection priority (seed=42, random within each tier):
  1. Score:True,  no blank answer
  2. Score:True,  blank answer
  3. Score:False, no blank answer
  4. Score:False, blank answer

"Blank answerer" = a run that contains a gpt-4o request with agent==7 (Answerer) and num_decode_tokens==0.

Writes selected_set.txt (one stem per line).
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

BASE        = Path(__file__).parent          # GAIATrace/owl/traces
DATA        = BASE.parent                    # GAIATrace/owl
TRACES_DIR  = BASE / "session_traces"
OUTCOMES    = BASE / "run_outcomes.csv"   # per-run solved/not verdict
MANIFEST    = BASE / "selected_set.txt"
SEED        = 42

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


def has_blank_answerer(csv_path):
    """True if any gpt-4o answerer (agent==7) row has decode==0."""
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("tokens", "").startswith("[200006, 77944"):
                continue
            try:
                if int(row["num_decode_tokens"]) == 0 and int(row.get("agent", -1)) == 7:
                    return True
            except (ValueError, KeyError):
                continue
    return False


def question_key(stem):
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem


def _tier(stem):
    """0 = best … 3 = worst."""
    sc    = get_score(stem)
    blank = has_blank_answerer(TRACES_DIR / f"{stem}.csv")
    if sc is True  and not blank: return 0
    if sc is True  and blank:     return 1
    if sc is not True and not blank: return 2
    return 3


def select_set(rng):
    all_stems = [
        f.stem for f in sorted(
            f for f in TRACES_DIR.glob("*.csv")
            if not f.stem.endswith("_tools")
        )
    ]
    groups = defaultdict(list)
    for s in all_stems:
        groups[question_key(s)].append(s)

    chosen = []
    tier_counts = [0, 0, 0, 0]

    for qkey in sorted(groups):
        candidates = groups[qkey]
        if len(candidates) == 1:
            chosen.append(candidates[0])
            tier_counts[_tier(candidates[0])] += 1
            continue

        best_tier = min(_tier(s) for s in candidates)
        best = [s for s in candidates if _tier(s) == best_tier]
        picked = rng.choice(best)
        chosen.append(picked)
        tier_counts[best_tier] += 1

    total_true = sum(1 for s in chosen if get_score(s) is True)
    print(f"Set: {len(chosen)} questions  ({total_true} Score:True  |  "
          f"tier breakdown: "
          f"T0(true+ok)={tier_counts[0]}  "
          f"T1(true+blank)={tier_counts[1]}  "
          f"T2(false+ok)={tier_counts[2]}  "
          f"T3(false+blank)={tier_counts[3]})")
    return chosen


def main():
    rng   = random.Random(SEED)
    stems = select_set(rng)

    MANIFEST.write_text("\n".join(stems) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
