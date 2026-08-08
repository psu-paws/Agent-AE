"""Shared pieces for the TTFT / TPOT CDF plots.

Run directory naming
--------------------
A run directory is named "<model_0>-<model_1>[-<policy>]-<qps>", where each token
PxDy is that pool's total prefill / decode tensor-parallel width (verified against
data/replica_groups_configs/*.json).  For the Miro targets model_0 is the SubLLM
pool and model_1 the MainLLM pool; the Owl targets store them the other way round.
Panel labels are written MainLLM-first.
"""
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

FS = 30
MAIN = "Main-LLM"
SUB = "Sub-LLM"

MODEL_CMAPS = [
    ["#253494", "#2c7fb8", "#41b6c4", "#6baed6"],  # Sub-LLM shades
    ["#006837", "#31a354", "#78c679", "#bae4bc"],  # Main-LLM shades
]
RUN_TYPE0_COLORS = ["#2c7fb8", "#006837"]

SJF = "$\\mathrm{SJF}$"
SJF60 = "$\\mathrm{SJF}_{60}$"
SJF180 = "$\\mathrm{SJF}_{180}$"
SJF60_S = "$\\mathrm{SJF}_{60}\\mathrm{-SFCFS}$"
SJF180_S = "$\\mathrm{SJF}_{180}\\mathrm{-SFCFS}$"

# Keyed by directory name; bold marks the widened dimension.
PLACEMENT_LABELS = {
    "4141-4141": "MainLLM(P4D4)-SubLLM(P4D4)",
    "4141-2321": "MainLLM($\\bf{P6}$D2)-SubLLM(P4D4)",
    "2181-4121": "MainLLM(P4D2)-SubLLM(P2$\\bf{D8}$)",
    "2141-8121": "MainLLM($\\bf{P8}$D2)-SubLLM(P2D4)",
    "2181-2141": "MainLLM(P2D4)-SubLLM(P2$\\bf{D8}$)",
    "4121-8121": "MainLLM($\\bf{P8}$D2)-SubLLM(P4D2)",
    "2321-4141": "MainLLM(P4D4)-SubLLM($\\bf{P6}$D2)",
    "8121-4121": "MainLLM(P4D2)-SubLLM($\\bf{P8}$D2)",
    "8121-2141": "MainLLM(P2D4)-SubLLM($\\bf{P8}$D2)",
    "4121-2181": "MainLLM(P2$\\bf{D8}$)-SubLLM(P4D2)",
}


def at_qps(runs, qps):
    """Bind (label, dir) pairs to one QPS suffix."""
    return [(label, "{target}/" + f"{d}-{qps}") for label, d in runs]


def placements(dirs, qps):
    """Placement runs labelled from PLACEMENT_LABELS."""
    return at_qps([(PLACEMENT_LABELS[d], d) for d in dirs], qps)


def swap_roles(label, target):
    """Owl stores model_0 as MainLLM; the label tables assume Miro's opposite."""
    if not target.startswith("Owl"):
        return label
    m = re.fullmatch(r"MainLLM\(([^)]+)\)-SubLLM\(([^)]+)\)(.*)", label, re.S)
    if m:
        return f"MainLLM({m.group(2)})-SubLLM({m.group(1)}){m.group(3)}"
    return re.sub(r"MainLLM|SubLLM",
                  lambda t: "SubLLM" if t.group() == "MainLLM" else "MainLLM",
                  label)


def model_style(target):
    """(names, cmaps, type0_colors) keyed by model_id."""
    if target.startswith("Owl"):
        return ({0: MAIN, 1: SUB},
                [MODEL_CMAPS[1], MODEL_CMAPS[0]],
                [RUN_TYPE0_COLORS[1], RUN_TYPE0_COLORS[0]])
    return {0: SUB, 1: MAIN}, MODEL_CMAPS, RUN_TYPE0_COLORS


def resolve_runs(runs_by_type, root, target, run_type):
    """Expand a run type into (label, run_dir) pairs, failing on missing inputs."""
    base = os.path.join(root, target)
    runs = [(swap_roles(label, target), path.format(target=base))
            for label, path in runs_by_type[run_type]]
    missing = [d for _, d in runs
               if not os.path.exists(os.path.join(d, "request_metrics.csv"))]
    if missing:
        raise SystemExit("Missing request_metrics.csv in:\n  " + "\n  ".join(missing))
    return runs


def _resolve_path(path, run_dir):
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for base in (os.getcwd(), project_root, os.path.dirname(os.path.abspath(run_dir))):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return None


def load_sessions(run_dir):
    """{turn_id: session_id}, from the trace when it carries one, else the plot CSV.

    Turn ids are opaque, so the mapping has to be read rather than derived.
    """
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        trace_file = (cfg.get("request_generator_config", {})
                         .get("length_generator_config", {})
                         .get("trace_file"))
        resolved = _resolve_path(trace_file, run_dir) if trace_file else None
        if resolved is not None:
            # pandas, not csv: the trace carries a token_ids column far past
            # csv's field size limit, and only two columns are needed.
            columns = pd.read_csv(resolved, nrows=0).columns
            if {"request_id", "session_id"}.issubset(columns):
                trace = pd.read_csv(resolved, usecols=["request_id", "session_id"])
                trace = trace.drop_duplicates("request_id")
                return dict(zip(trace["request_id"].astype(int),
                                trace["session_id"].astype(int)))

    requestwise = os.path.join(run_dir, "plots", "request_e2e_time_requestwise.csv")
    if not os.path.exists(requestwise):
        raise SystemExit(f"Cannot map turns to sessions: no trace session_id and "
                         f"{requestwise} is missing")
    with open(requestwise) as f:
        return {int(r["Request Id"]): int(float(r["request"])) for r in csv.DictReader(f)}


def load_turns(run_dir):
    """(ttft, tpot, model) dicts keyed by turn id.

    TTFT is prefill_e2e_time (queue + prefill from arrival). TPOT is the normalized
    decode time, None for turns that decoded nothing.
    """
    ttft, tpot, model = {}, {}, {}
    with open(os.path.join(run_dir, "request_metrics.csv")) as f:
        for row in csv.DictReader(f):
            rid = int(row["Request Id"])
            if row.get("prefill_e2e_time"):
                ttft[rid] = float(row["prefill_e2e_time"])
            decode = row.get("decode_time_execution_plus_preemption_normalized", "")
            if decode:
                n_decode = int(float(row.get("request_num_decode_tokens", "0")))
                tpot[rid] = float(decode) if n_decode > 0 else None
            if row.get("model_id"):
                model[rid] = int(float(row["model_id"]))
    return ttft, tpot, model


def per_request(values, model):
    """{model_id: array} at turn granularity."""
    grouped = defaultdict(list)
    for rid, value in values.items():
        if value is not None and rid in model:
            grouped[model[rid]].append(value)
    return {m: np.array(v) for m, v in sorted(grouped.items())}


def per_session(values, model, sessions, agg):
    """{model_id: array} of per-session aggregates (agg is sum or mean)."""
    grouped = defaultdict(lambda: defaultdict(list))
    for rid, value in values.items():
        if value is not None and rid in model and rid in sessions:
            grouped[model[rid]][sessions[rid]].append(value)
    return {m: np.array([agg(v) for v in s.values() if v])
            for m, s in sorted(grouped.items())}


def plot_cdf(ax, entries, xlabel, show_ylabel=True, extend_left=False):
    """Draw one CDF per entry, where an entry is (array, label, color, linestyle)."""
    for arr, label, color, ls in entries:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=label, color=color, lw=5, linestyle=ls)

    # Capture the autoscaled range first so the extensions below cannot widen it.
    xmin, xmax = ax.get_xlim()
    for arr, _, color, ls in entries:
        ax.plot([np.max(arr), xmax], [1.0, 1.0], color=color, lw=5, linestyle=ls)
        if extend_left:
            x0 = float(np.min(arr))
            ax.plot([max(0.0, xmin), x0, x0], [0.0, 0.0, 1.0 / len(arr)],
                    color=color, lw=5, linestyle=ls)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(xlabel, fontsize=FS - 2)
    if show_ylabel:
        ax.set_ylabel("CDF", fontsize=FS - 2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=FS - 4)
    ax.grid(True, alpha=0.3)


def print_stats(rows, width=28):
    """rows: (tag, array) pairs."""
    print(f"\n{'':>{width}}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'mean':>8}  {'n':>6}")
    for tag, arr in rows:
        stats = (np.percentile(arr, 50), np.percentile(arr, 95),
                 np.percentile(arr, 99), arr.mean())
        print(f"{tag:>{width}}  " + "  ".join(f"{v:>8.3f}" for v in stats)
              + f"  {len(arr):>6}")
