"""
Per-request Gantt-style stacked bar comparison.

Each panel shows one request group per bar, segments summed per type across
all turns, sorted by total latency ascending.

Layout auto-sizes: 1 run -> 1x1, 2 runs -> 1x2, 3 runs -> 1x3, 4+ -> grid.

Run directory naming
--------------------
A placement run directory is named "<model_0>-<model_1>[-<policy>]-<qps>", where
each token PxDy is that pool's total prefill / decode tensor-parallel width
(verified against data/replica_groups_configs/*.json).  model_0 is the SubLLM
pool and model_1 the MainLLM pool, so a directory "8121-4121" is
SubLLM(P8D2) with MainLLM(P4D2).  Panel titles are written MainLLM-first.
"""
import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

SJF = "$\\mathrm{SJF}$"
SJF60 = "$\\mathrm{SJF}_{60}$"
SJF180 = "$\\mathrm{SJF}_{180}$"
SJF60_S = "$\\mathrm{SJF}_{60}\\mathrm{-SFCFS}$"
SJF180_S = "$\\mathrm{SJF}_{180}\\mathrm{-SFCFS}$"
PREFILL_SUB_UP = "$\\mathrm{Prefill}_{\\mathrm{SubLLM}}$↑"
DECODE_MAIN_DN = "$\\mathrm{Decode}_{\\mathrm{MainLLM}}$↓"
PREFILL_MAIN_DN = "$\\mathrm{Prefill}_{\\mathrm{MainLLM}}$↓"

# Scheduling-policy sweep on a fixed balanced placement.
POLICY_RUNS = [
    ("FCFS",     "4141-4141"),
    ("T-FCFS",   "4141-4141-SFCFS"),
    (SJF60,      "4141-4141-SJF60"),
    (SJF60_S,    "4141-4141-SJF60-SFCFS"),
    (SJF180,     "4141-4141-SJF180"),
    (SJF180_S,   "4141-4141-SJF180-SFCFS"),
    (SJF,        "4141-4141-SJF"),
]

# Subset of the policy sweep that was actually run at QPS0.5.
POLICY_RUNS_QPS05 = [
    (SJF,      "4141-4141-SJF"),
    (SJF180,   "4141-4141-SJF180"),
    ("T-FCFS", "4141-4141-SFCFS"),
]

# Placement sweep A.  Bold marks the widened dimension relative to balanced.
PLACEMENT_RUNS_A = [
    ("MainLLM(P4D4)-SubLLM(P4D4)",         "4141-4141"),
    ("MainLLM($\\bf{P6}$D2)-SubLLM(P4D4)", "4141-2321"),
    ("MainLLM(P4D2)-SubLLM(P2$\\bf{D8}$)", "2181-4121"),
    ("MainLLM($\\bf{P8}$D2)-SubLLM(P2D4)", "2141-8121"),
    ("MainLLM(P2D4)-SubLLM(P2$\\bf{D8}$)", "2181-2141"),
    ("MainLLM($\\bf{P8}$D2)-SubLLM(P4D2)", "4121-8121"),
]

# Placement sweep B (mirrored directory names).
PLACEMENT_RUNS_B = [
    ("MainLLM(P4D4)-SubLLM(P4D4)",         "4141-4141"),
    ("MainLLM(P4D4)-SubLLM($\\bf{P6}$D2)", "2321-4141"),
    ("MainLLM(P4D2)-SubLLM($\\bf{P8}$D2)", "8121-4121"),
    ("MainLLM(P2D4)-SubLLM($\\bf{P8}$D2)", "8121-2141"),
    ("MainLLM(P2$\\bf{D8}$)-SubLLM(P4D2)", "4121-2181"),
]

# Prefill/decode imbalance study.
IMBALANCE_RUNS = [
    ("Balanced",                                  "4141-4141"),
    (f"{PREFILL_SUB_UP}\n{DECODE_MAIN_DN}",       "8121-4121"),
    (f"{PREFILL_SUB_UP}\n{PREFILL_MAIN_DN}",      "8121-2141"),
]


def at_qps(runs, qps):
    """Bind a run set to one QPS suffix, producing (label, path-template) pairs."""
    return [(label, "{target}/" + f"{d}-{qps}") for label, d in runs]


def swap_roles(label, target):
    """Owl stores model_0 as MainLLM; the labels above assume Miro's opposite."""
    if not target.startswith("Owl"):
        return label
    m = re.fullmatch(r"MainLLM\(([^)]+)\)-SubLLM\(([^)]+)\)(.*)", label, re.S)
    if m:
        return f"MainLLM({m.group(2)})-SubLLM({m.group(1)}){m.group(3)}"
    return re.sub(r"MainLLM|SubLLM",
                  lambda t: "SubLLM" if t.group() == "MainLLM" else "MainLLM",
                  label)


RUNS_BY_TYPE = {
    0: [("No prefix cache", "{target}/4141-4141-NoPC"),
        ("prefix cache",    "{target}/4141-4141-QPS01")],
    1: at_qps(POLICY_RUNS, "QPS01"),
    2: at_qps(POLICY_RUNS, "QPS005"),
    3: at_qps(POLICY_RUNS_QPS05, "QPS05"),
    4: at_qps(PLACEMENT_RUNS_A, "QPS01"),
    5: at_qps(PLACEMENT_RUNS_A, "QPS005"),
    6: at_qps(PLACEMENT_RUNS_A, "QPS05"),
    7: at_qps(IMBALANCE_RUNS, "QPS01"),
    8: at_qps(PLACEMENT_RUNS_B, "QPS005"),
    9: at_qps(IMBALANCE_RUNS, "QPS05"),
    10: [("TPS0.05", "{target}/4141-4141-QPS005"),
         ("TPS0.5",  "{target}/4141-4141-QPS05")],
    # Validation setups, all six simulated replays behind the fidelity table.
    # Target is "validate". S1-S4 at 0.1 sessions/s plus S1 at 0.04 and 0.01.
    11: [("S1 8B TP-2",     "{target}/S1-QPS01"),
         ("S2 8B TP-1",     "{target}/S2-QPS01"),
         ("S3 32B TP-2",    "{target}/S3-QPS01"),
         ("S4 8B 2x(TP-1)", "{target}/S4-QPS01"),
         ("S1 @0.04",       "{target}/S1-QPS004"),
         ("S1 @0.01",       "{target}/S1-QPS001")],
}

M0 = "MainLLM"
M1 = "SubLLM"
FS = 30
STAT_COLORS = {"p50": "#ff6600", "p90": "#ff0000"}

YLIM_OVERRIDES: dict[tuple[str, int], float] = {}
LEGEND_Y = {("Miro", 10): 0.94}


def build_segments(target):
    """Segment table: (label, csv_key, color, hatch).

    model_0 is the SubLLM pool for the Miro targets and the MainLLM pool for the
    Owl targets, so the two model columns swap roles between them.
    """
    sub_first = not target.startswith("Owl")
    sub, main = ("0", "1") if sub_first else ("1", "0")
    return [
        (f"{M1} Queue",   f"model_{sub}_queue_wait",       "#253494", "X"),
        (f"{M1} Prefill", f"model_{sub}_prefill_latency",  "#2c7fb8", ""),
        (f"{M1} Decode",  f"model_{sub}_decode_latency",   "#41b6c4", ""),
        (f"{M0} Queue",   f"model_{main}_queue_wait",      "#006837", ""),
        (f"{M0} Prefill", f"model_{main}_prefill_latency", "#31a354", ""),
        (f"{M0} Decode",  f"model_{main}_decode_latency",  "#78c679", "//"),
        ("Tool",          "inter_request_latency",         "#252525", ""),
    ]


def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            # CSVs built before the rename call the session column request_id.
            key = "session_id" if "session_id" in row else "request_id"
            rid = int(row[key])
            rows[rid] = {k: float(v) for k, v in row.items() if k != key}
            rows[rid].setdefault("inter_request_latency", 0.0)
    return rows


def total(row, segments):
    return sum(row.get(key, 0.0) for _, key, *_ in segments)


def draw_stat_lines(ax, latencies, n_bars, ylim, cfg, idx=0):
    stats = [("p50", np.percentile(latencies, 50), "--"),
             ("p90", np.percentile(latencies, 90), "-.")]
    x = n_bars * 0.01
    for name, val, ls in stats:
        color = STAT_COLORS[name]
        ax.axhline(val, color=color, linestyle=ls, linewidth=3, alpha=0.85, zorder=5)
        off = 12 if cfg.run_type in (7, 9) else 30
        va = "bottom"
        if cfg.run_type == 9 and idx == 2 and name == "p90":
            off = 0.13 * ylim  # pull the p90 label well down (fraction of axis)
        elif name == "p90" and cfg.target == "Miro_70B" and idx in (1, 2):
            off = 0.02 * ylim  # small gap; text hangs below the line
            va = "top"
        ax.text(x, val - off, f"{name}={val:.0f}s",
                fontsize=FS - 4, color=color, va=va, ha="left", zorder=6)


def draw_gantt(ax, data, order, title, cfg, ylim, show_ylabel, show_xlabel=True, idx=0):
    n = len(order)
    x = np.arange(n)
    bottoms = np.zeros(n)

    for seg_label, key, color, hatch in cfg.segments:
        vals = np.array([data[r].get(key, 0.0) for r in order])
        ax.bar(x, vals, bottom=bottoms, color=color, hatch=hatch, label=seg_label,
               width=1.0, linewidth=0)
        bottoms += vals

    latencies = np.array([total(data[r], cfg.segments) for r in order])
    draw_stat_lines(ax, latencies, n, ylim, cfg, idx=idx)

    if cfg.run_type in (7, 9):
        ax.set_title(title, fontsize=FS - 2, loc="left",
                     x=0.1 if idx == 1 else 0.2, y=0.95, va="top", ma="left")
    else:
        ax.set_title(title, fontsize=FS + 2, y=0.84)
    if show_xlabel:
        ax.set_xlabel("Tasks sorted by latency", fontsize=FS - 2)
    if show_ylabel:
        ax.set_ylabel("Latency (s)", fontsize=FS - 2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, ylim)
    ax.tick_params(labelsize=FS - 4)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)


def build_legend(fig, axes_flat, cfg):
    """Two-row legend: Main-LLM / Sub-LLM rows over Queue|Prefill|Decode, plus Tool.

    matplotlib fills a multi-column legend column by column, so for 9 entries in
    5 columns it uses rows_per_col = [2, 2, 2, 2, 1]; order the flat list to match.
    """
    handles, labels = axes_flat[0].get_legend_handles_labels()
    hmap = dict(zip(labels, handles))
    blank = lambda: Patch(facecolor="none", edgecolor="none")
    entries = [
        (blank(), "Main-LLM:"),          (blank(), "Sub-LLM:"),
        (hmap[f"{M0} Queue"], "Queue"),  (hmap[f"{M1} Queue"], "Queue"),
        (hmap[f"{M0} Prefill"], "Prefill"), (hmap[f"{M1} Prefill"], "Prefill"),
        (hmap[f"{M0} Decode"], "Decode"),   (hmap[f"{M1} Decode"], "Decode"),
        (hmap["Tool"], "Tool"),
    ]
    legend_y = LEGEND_Y.get((cfg.target, cfg.run_type),
                            0.90 if cfg.run_type == 7 else 0.93)
    leg = fig.legend(*zip(*entries), loc="upper center", ncol=5,
                     fontsize=FS - 2, bbox_to_anchor=(0.5, legend_y),
                     frameon=True, borderpad=0.2, labelspacing=0.2, handlelength=1.0,
                     handleheight=0.7, handletextpad=0.3, columnspacing=0.8,
                     borderaxespad=0.1)
    # Vertically center single-entry columns (the lone "Tool") between the two rows.
    hbox = getattr(leg, "_legend_handle_box", None)
    if hbox is not None:
        hbox.align = "center"


def resolve_layout(n, ncols_arg):
    """Return (nrows, ncols, fig_width). Three panels stay as wide as two."""
    if n == 3 and not ncols_arg:
        return 1, 3, 6.5 * 2
    ncols = ncols_arg if ncols_arg else (3 if n >= 6 else min(n, 2))
    return (n + ncols - 1) // ncols, ncols, 6.5 * ncols


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="Miro")
    parser.add_argument("--root", default="simulator_output",
                        help="Directory containing the target folders")
    parser.add_argument("--run-type", type=int, default=0, choices=sorted(RUNS_BY_TYPE))
    parser.add_argument("--ncols", type=int, default=None)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    cfg = argparse.Namespace(target=args.target, run_type=args.run_type,
                             segments=build_segments(args.target))

    base = os.path.join(args.root, args.target)
    runs = [(swap_roles(label, args.target), path.format(target=base))
            for label, path in RUNS_BY_TYPE[args.run_type]]

    missing = [d for _, d in runs
               if not os.path.exists(os.path.join(d, "request_group_latency_by_model.csv"))]
    if missing:
        raise SystemExit("Missing request_group_latency_by_model.csv in:\n  "
                         + "\n  ".join(missing))

    datasets = [(label, load_csv(os.path.join(d, "request_group_latency_by_model.csv")))
                for label, d in runs]
    common = sorted(set.intersection(*[set(data) for _, data in datasets]))
    if not common:
        raise SystemExit("No request_id is present in every run of this comparison.")

    orders = {label: sorted(common, key=lambda r: total(data[r], cfg.segments))
              for label, data in datasets}
    data_max = max(total(data[r], cfg.segments) for _, data in datasets for r in common)
    ylim = YLIM_OVERRIDES.get((args.target, args.run_type), data_max * 1.05)
    if ylim < data_max:
        print(f"[warn] ylim override {ylim:g} clips data reaching {data_max:.1f}s")

    n = len(datasets)
    nrows, ncols, fig_w = resolve_layout(n, args.ncols)
    three_up = n == 3 and not args.ncols

    # The validation panels span an order of magnitude: S4 runs a 32B model and
    # dwarfs the 8B setups, so one shared axis flattens the other five into the
    # baseline. Give S4 its own scale and share the rest.
    per_panel_ylim = None
    if args.run_type == 11:
        outlier = [i for i, (lab, _) in enumerate(datasets) if lab.startswith("S4")]
        rest = [max(total(data[r], cfg.segments) for r in common)
                for i, (_, data) in enumerate(datasets) if i not in outlier]
        shared = max(rest) * 1.05
        per_panel_ylim = [
            (max(total(data[r], cfg.segments) for r in common) * 1.05
             if i in outlier else shared)
            for i, (_, data) in enumerate(datasets)]

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 5 * nrows),
                             sharey=per_panel_ylim is None)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (label, data) in enumerate(datasets):
        draw_gantt(axes_flat[idx], data, orders[label], label, cfg,
                   ylim if per_panel_ylim is None else per_panel_ylim[idx],
                   show_ylabel=(idx % ncols == 0) or per_panel_ylim is not None,
                   show_xlabel=(idx == 1) if three_up else True,
                   idx=idx)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    build_legend(fig, axes_flat, cfg)

    out_path = os.path.join(args.outdir, f"gantt_{args.target}_type{args.run_type}.pdf")
    fig.tight_layout(rect=[0, 0, 1, 0.74])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    col_w = 10
    totals = [(label, [total(data[r], cfg.segments) for r in common]) for label, data in datasets]
    print(f"\n{'':>12}" + "".join(f"  {label:>{col_w}}" for label, _ in totals))
    for stat, p in [("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99), ("mean", None)]:
        vals = [np.percentile(arr, p) if p else np.mean(arr) for _, arr in totals]
        print(f"{stat:>12}" + "".join(f"  {v:>{col_w}.2f}" for v in vals))


if __name__ == "__main__":
    main()
