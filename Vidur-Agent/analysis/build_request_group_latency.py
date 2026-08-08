"""
Build a per-session latency CSV from a single simulator output directory.

Reads request_metrics.csv (under PD a turn has two rows, prefill and decode) and
writes one row per session: a prefill/decode/queue triple per model, plus
inter_request_latency.

Turns released together form a *level* -- from the trace's `dep` column, or one
turn per level when there is none.  Each level is decomposed from its longest turn,
whose span is the level's:

  prefill = its prefill_time_execution_plus_preemption
  decode  = its completion − its prefill completion
  queue   = level span − prefill − decode, floored at its scheduling delay
  inter_request_latency = session span − Σ level spans

Concurrent siblings are not added in: their prefill overlaps that turn's decode, so
summing would double-count wall-clock.  Their prefill lands in queue instead.

Usage:
  python analysis/build_request_group_latency.py <run_dir> [output_csv]
"""

import argparse
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd

METRIC_COLS = ["prefill_latency", "decode_latency", "queue_wait"]


# ── config lookups ─────────────────────────────────────────────────────────────

def _resolve_path(path: str, run_dir: str) -> str | None:
    """Resolve a config-relative path against cwd / project root / run_dir parent."""
    if os.path.isabs(path):
        return path if os.path.exists(path) else None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in (os.getcwd(), project_root, os.path.dirname(os.path.abspath(run_dir))):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_config(run_dir: str) -> dict:
    config_json = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_json):
        return {}
    with open(config_json) as f:
        return json.load(f)


def load_replica_model_map(run_dir: str) -> dict:
    """Return {replica_id: model_id} from replica_groups_pools in config.json."""
    cfg = _load_config(run_dir)
    groups_config_path = cfg.get("cluster_config", {}).get("replica_groups_config")
    if not groups_config_path:
        return {}

    resolved = _resolve_path(groups_config_path, run_dir)
    if resolved is None:
        print("[warn] replica_groups_config not found; defaulting all replicas to model 0",
              file=sys.stderr)
        return {}

    with open(resolved) as f:
        groups_cfg = json.load(f)

    replica_model_map = {}
    for model_id, pool_def in enumerate(groups_cfg.get("replica_groups_pools", [])):
        for role in ("prefill", "decode"):
            for replica in pool_def.get(role, []):
                replica_model_map[int(replica)] = model_id
    return replica_model_map


def _trace_path(run_dir: str) -> str | None:
    """Absolute path to the trace this run was driven by, or None."""
    cfg = _load_config(run_dir)
    trace_file = (
        cfg.get("request_generator_config", {})
           .get("length_generator_config", {})
           .get("trace_file")
    )
    return _resolve_path(trace_file, run_dir) if trace_file else None


def load_turn_sessions(run_dir: str) -> pd.DataFrame | None:
    """Map each turn to its session from the trace, or None if it cannot.

    The simulator uses the trace's request_id verbatim as its "Request Id", so
    this is exact rather than inferred.
    """
    resolved = _trace_path(run_dir)
    if resolved is None:
        return None
    if not {"request_id", "session_id"}.issubset(pd.read_csv(resolved, nrows=0).columns):
        return None
    trace = pd.read_csv(resolved, usecols=["request_id", "session_id"])
    return trace.drop_duplicates("request_id").rename(columns={"request_id": "Request Id"})


def load_dep_levels(run_dir: str) -> dict:
    """Return {request_id: dependency depth} from the trace's dep column.

    `dep` lists the turn_ids that must finish first, so a turn's depth is the
    longest chain reaching it and turns sharing a depth are released together.
    Empty when the trace has no dep column, making every session sequential.
    """
    resolved = _trace_path(run_dir)
    if resolved is None:
        return {}

    needed = ["dep", "session_id", "request_id"]
    header = pd.read_csv(resolved, nrows=0).columns
    if not set(needed).issubset(header):
        return {}
    if "turn_id" in header:
        needed.append("turn_id")
    trace = pd.read_csv(resolved, usecols=needed)
    if "turn_id" not in trace.columns:
        # Position within the session stands in for a missing turn_id.
        trace["turn_id"] = trace.groupby("session_id").cumcount()

    levels: dict = {}
    for _, session in trace.groupby("session_id", sort=False):
        deps = {}
        for row in session.itertuples():
            raw = row.dep
            parsed = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
            deps[int(row.turn_id)] = [int(d) for d in parsed]

        depth: dict = {}

        def depth_of(turn_id: int) -> int:
            if turn_id in depth:
                return depth[turn_id]
            # A dep on a turn outside this session is ignored.
            parents = [d for d in deps.get(turn_id, []) if d in deps]
            depth[turn_id] = 1 + max((depth_of(p) for p in parents), default=-1)
            return depth[turn_id]

        for row in session.itertuples():
            levels[int(row.request_id)] = depth_of(int(row.turn_id))
    return levels


# ── loading ────────────────────────────────────────────────────────────────────

def load_metrics(run_dir: str) -> pd.DataFrame:
    """Load request_metrics.csv with each turn labelled by its session."""
    metrics_path = os.path.join(run_dir, "request_metrics.csv")
    if not os.path.exists(metrics_path):
        sys.exit(f"Error: {metrics_path} not found")
    metrics = pd.read_csv(metrics_path)

    sessions = load_turn_sessions(run_dir)
    if sessions is None:
        # Fall back to the requestwise plot CSV, whose "request" column is the
        # session the simulator grouped by. A byproduct, hence second choice.
        requestwise_path = os.path.join(
            run_dir, "plots", "request_e2e_time_requestwise.csv"
        )
        if not os.path.exists(requestwise_path):
            sys.exit(
                "Error: cannot map turns to sessions — the trace has no "
                f"request_id/session_id columns and {requestwise_path} is missing"
            )
        sessions = (
            pd.read_csv(requestwise_path)[["Request Id", "request"]]
            .drop_duplicates()
            .rename(columns={"request": "session_id"})
        )

    metrics = metrics.merge(sessions, on="Request Id", how="left")

    missing = int(metrics["session_id"].isna().sum())
    if missing:
        print(f"Warning: {missing} rows have no matching session; dropped.", file=sys.stderr)
        metrics = metrics.dropna(subset=["session_id"])
    metrics = metrics.reset_index(drop=True)
    metrics["session_id"] = metrics["session_id"].astype(int)

    # model_id is recorded only on decode-side rows; spread it across the turn.
    if "model_id" in metrics.columns:
        turn_model = (
            metrics.dropna(subset=["model_id"]).groupby("Request Id")["model_id"].first()
        )
        metrics["model_id"] = metrics["Request Id"].map(turn_model)

    return metrics


# ── grouping ───────────────────────────────────────────────────────────────────

def add_model_column(metrics: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Ensure every row carries the model_id its request targeted."""
    if "model_id" in metrics.columns and metrics["model_id"].notna().any():
        return metrics

    # No model_id recorded: fall back to whichever model each replica serves.
    metrics = metrics.copy()
    replica_model_map = load_replica_model_map(run_dir)
    if not replica_model_map:
        print("[warn] No model_id column; all requests → model 0", file=sys.stderr)
    metrics["model_id"] = (
        metrics["replica"].astype(int).map(replica_model_map).fillna(0).astype(int)
    )
    return metrics


# ── turns ──────────────────────────────────────────────────────────────────────

def collapse_to_turns(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per turn, folding PD's prefill and decode rows back together."""
    turns = metrics.groupby("Request Id").agg(
        session_id=("session_id", "first"),
        model=("model_id", "first"),
        arrived_at=("request_arrived_at", "min"),
        prefill_work=("prefill_time_execution_plus_preemption", "sum"),
        scheduling_delay=("request_scheduling_delay", "min"),
        prefill_e2e=("prefill_e2e_time", "max"),
        e2e=("request_e2e_time", "max"),
    )
    turns["prefill_e2e"] = turns["prefill_e2e"].fillna(0.0)
    turns["e2e"] = turns["e2e"].fillna(turns["prefill_e2e"])
    turns["prefill_work"] = turns["prefill_work"].fillna(0.0)
    turns["scheduling_delay"] = turns["scheduling_delay"].fillna(0.0)

    turns["start"] = turns["arrived_at"]
    turns["end"] = turns["arrived_at"] + turns["e2e"]
    turns["prefill_done_at"] = turns["arrived_at"] + turns["prefill_e2e"]
    return turns.reset_index()


# ── levels ─────────────────────────────────────────────────────────────────────

def _charge_level(members: list, session_id: int,
                  phase_rows: list, spans: list, diag: dict) -> None:
    """Charge a level: every turn's prefill, the longest turn's decode, rest queue."""
    span = max(0.0, max(m.end for m in members) - min(m.start for m in members))
    critical = max(members, key=lambda m: m.end - m.start)

    # Siblings are not summed in: their prefill overlaps this turn's decode.
    prefill = critical.prefill_work
    decode = max(0.0, critical.end - critical.prefill_done_at)
    residual = span - prefill - decode

    if residual < critical.scheduling_delay:
        # Queue can never be under the measured delay; shrink prefill to fit.
        queue = critical.scheduling_delay
        shrunk = max(0.0, span - decode - queue)
        diag["n_queue_clamped"] += 1
        diag["prefill_trimmed_s"] += prefill - shrunk
        prefill = shrunk
    else:
        queue = residual

    diag["n_levels"] += 1
    if len(members) > 1:
        diag["n_multi_turn_levels"] += 1
        diag["n_concurrent_turns"] += len(members) - 1

    spans.append({"session_id": session_id, "span": span})
    phase_rows.append({
        "session_id": session_id, "model": critical.model,
        "prefill_latency": prefill, "decode_latency": decode, "queue_wait": queue,
    })


def build_levels(turns: pd.DataFrame, dep_levels: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Group each session's turns by dependency depth and charge each level."""
    turns = turns.copy()
    if dep_levels:
        turns["level"] = turns["Request Id"].map(dep_levels)
        unmapped = int(turns["level"].isna().sum())
        if unmapped:
            print(f"[warn] {unmapped} turns are absent from the trace's dep graph; "
                  "each treated as its own level.", file=sys.stderr)
            # Distinct negative levels so unmapped turns can never be merged.
            turns.loc[turns["level"].isna(), "level"] = -1 - np.arange(unmapped)
    else:
        turns["level"] = np.arange(len(turns))

    phase_rows: list = []
    spans: list = []
    diag = {"n_levels": 0, "n_multi_turn_levels": 0, "n_concurrent_turns": 0,
            "n_queue_clamped": 0, "prefill_trimmed_s": 0.0}

    for (session_id, _), members in turns.groupby(["session_id", "level"], sort=False):
        _charge_level(list(members.itertuples()), session_id, phase_rows, spans, diag)

    phases = pd.DataFrame(phase_rows)
    phases["model"] = phases["model"].astype(int)
    return phases, pd.DataFrame(spans), diag


def pivot_wide(phases: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Aggregate per (session, model) and pivot to one row per session."""
    per_model = phases.groupby(["session_id", "model"])[METRIC_COLS].sum()
    models = sorted(per_model.index.get_level_values("model").unique())
    col_order = [f"model_{m}_{metric}" for metric in METRIC_COLS for m in models]

    wide = per_model.unstack("model")
    wide.columns = [f"model_{m}_{metric}" for metric, m in wide.columns]
    out_df = wide.reindex(columns=col_order).fillna(0.0).reset_index()
    return out_df, models


# ── verification ───────────────────────────────────────────────────────────────

def write_verification(path: str, out_df: pd.DataFrame, session_first: pd.Series,
                       session_last: pd.Series, component_cols: list) -> None:
    """Write a per-session table of session duration vs. summed latency components."""
    indexed = out_df.set_index("session_id")
    session_duration = (session_last - session_first).clip(lower=0.0)
    sum_components = indexed[component_cols].sum(axis=1) + indexed["inter_request_latency"]

    header = (f"{'session_id':>10}  {'arrived_at':>12}  {'completed_at':>13}"
              f"  {'session_duration':>16}  {'sum_components':>14}  {'diff':>10}")
    rule = "-" * len(header)

    diffs = []
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write(rule + "\n")
        for session_id in indexed.index:
            arrived = session_first.get(session_id, float("nan"))
            completed = session_last.get(session_id, float("nan"))
            duration = session_duration.get(session_id, float("nan"))
            total = sum_components.get(session_id, float("nan"))
            diffs.append(total - duration)
            f.write(f"{session_id:>10}  {arrived:>12.4f}  {completed:>13.4f}"
                    f"  {duration:>16.4f}  {total:>14.4f}  {diffs[-1]:>10.6f}\n")
        f.write(rule + "\n")
        if diffs:
            f.write(f"\nmax |diff| = {max(abs(d) for d in diffs):.6f} s\n")
            f.write(f"mean diff  = {statistics.mean(diffs):.6f} s\n")
            f.write(f"p50  diff  = {statistics.median(diffs):.6f} s\n")


# ── core builder ───────────────────────────────────────────────────────────────

def build_latency_table(run_dir: str, output_csv: str) -> None:
    metrics = add_model_column(load_metrics(run_dir), run_dir)
    turns = collapse_to_turns(metrics)
    phases, spans, diagnostics = build_levels(turns, load_dep_levels(run_dir))
    out_df, models = pivot_wide(phases)

    session_first = turns.groupby("session_id")["start"].min()
    session_last = turns.groupby("session_id")["end"].max()
    session_e2e = spans.groupby("session_id")["span"].sum()

    # Time the levels do not explain is the gap between them.
    raw_inter = session_last - session_first - session_e2e
    overcounted = raw_inter < -1e-3      # well above float noise
    out_df["inter_request_latency"] = out_df["session_id"].map(raw_inter.clip(lower=0.0))

    # Re-tests the pivot, not the decomposition: the phases are constructed to
    # sum to the level span. The verify file's diff column is the real check.
    component_cols = [f"model_{m}_{metric}" for metric in METRIC_COLS for m in models]
    check = out_df[component_cols].sum(axis=1).to_numpy()
    expected = session_e2e.reindex(out_df["session_id"]).to_numpy()
    diff = np.abs(check - expected)
    max_diff = float(np.nanmax(diff)) if len(diff) and not np.all(np.isnan(diff)) else 0.0

    out_df.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"Written  : {output_csv}")
    print(f"Models   : {models}")
    print(f"Rows     : {len(out_df)}  sessions")
    print(f"Columns  : {len(out_df.columns)}")
    print(f"Levels   : {diagnostics['n_levels']}  "
          f"({diagnostics['n_multi_turn_levels']} held concurrent turns, "
          f"{diagnostics['n_concurrent_turns']} turns ran alongside another)")
    print(f"Clamped  : {diagnostics['n_queue_clamped']}/{diagnostics['n_levels']} levels had "
          f"queue raised to the measured scheduling delay "
          f"({diagnostics['prefill_trimmed_s']:.6f} s of prefill trimmed)")
    print(f"Bookkeep : max |Σphases − Σlevel spans| = {max_diff:.2e} s (pivot check)")
    if overcounted.any():
        print(f"[warn] {int(overcounted.sum())} sessions have level time exceeding their span "
              f"by up to {float(-raw_inter.min()):.4f} s (clipped to 0 inter-request latency).",
              file=sys.stderr)

    verify_path = os.path.join(os.path.dirname(output_csv), "request_group_latency_verify.txt")
    write_verification(verify_path, out_df, session_first, session_last, component_cols)
    print(f"Verify   : {verify_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build per-session latency CSV from a simulator run directory."
    )
    parser.add_argument(
        "run_dir",
        help="Path to a single run output folder (contains request_metrics.csv)",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        help="Output CSV path (default: <run_dir>/request_group_latency_by_model.csv)",
    )
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(
        args.run_dir, "request_group_latency_by_model.csv"
    )
    build_latency_table(args.run_dir, output_csv)


if __name__ == "__main__":
    main()
