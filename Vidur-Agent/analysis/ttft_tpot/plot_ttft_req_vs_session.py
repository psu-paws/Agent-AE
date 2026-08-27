"""
Side-by-side TTFT CDF for a single target.

  left  = per-request TTFT
  right = per-session TTFT (sum of per-turn TTFTs within each session)

Only the Sub-LLM series is drawn; sessions come from the trace, never from the
turn id. See _common.py for the run directory naming.

Usage:
  python analysis/ttft_tpot/plot_ttft_req_vs_session.py --target Miro --run-type 7
  python analysis/ttft_tpot/plot_ttft_req_vs_session.py --target Owl  --run-type 5
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np

import _common as C

RUN_LINESTYLES = ["-", "--", ":", "-."]

POLICY_RUNS = [
    ("FCFS",    "4141-4141"),
    ("SFCFS",   "4141-4141-SFCFS"),
    (C.SJF60,   "4141-4141-SJF60"),
    (C.SJF60_S, "4141-4141-SJF60-SFCFS"),
    (C.SJF180,  "4141-4141-SJF180"),
    (C.SJF180_S, "4141-4141-SJF180-SFCFS"),
    (C.SJF,     "4141-4141-SJF"),
]
POLICY_RUNS_QPS05 = [
    ("Q-FCFS",  "4141-4141"),
    (C.SJF,     "4141-4141-SJF"),
    (C.SJF180,  "4141-4141-SJF180"),
    ("T-FCFS",  "4141-4141-SFCFS"),
]
PLACEMENT_A = ["4141-4141", "4141-2321", "2181-4121", "2141-8121", "2181-2141", "4121-8121"]
PLACEMENT_B = ["4141-4141", "2321-4141", "8121-4121", "8121-2141", "4121-2181"]

RUNS_BY_TYPE = {
    0: [("NoPrefix", "{target}/4141-4141-NoPC"),
        ("Prefix",   "{target}/4141-4141-QPS01")],
    1: C.at_qps(POLICY_RUNS, "QPS01"),
    2: C.at_qps(POLICY_RUNS, "QPS005"),
    3: C.at_qps(POLICY_RUNS_QPS05, "QPS05"),
    4: C.placements(PLACEMENT_A, "QPS01"),
    5: C.placements(PLACEMENT_A, "QPS005"),
    6: C.placements(PLACEMENT_A, "QPS05"),
    7: C.placements(PLACEMENT_B, "QPS01"),
    8: C.placements(PLACEMENT_B, "QPS005"),
    9: C.placements(PLACEMENT_B, "QPS05"),
    10: [("QPS0.05", "{target}/4141-4141-QPS005"),
         ("QPS0.1",  "{target}/4141-4141-QPS01"),
         ("QPS0.3",  "{target}/4141-4141-QPS03"),
         ("QPS0.5",  "{target}/4141-4141-QPS05")],
    11: [(f"{C.PLACEMENT_LABELS['4141-4141']},QPS0.05", "{target}/4141-4141-QPS005"),
         (f"{C.PLACEMENT_LABELS['2181-4121']},QPS0.05", "{target}/2181-4121-QPS005"),
         (f"{C.PLACEMENT_LABELS['4141-4141']},QPS0.5",  "{target}/4141-4141-QPS05"),
         (f"{C.PLACEMENT_LABELS['4141-2321']},QPS0.5",  "{target}/4141-2321-QPS05")],
}


def build_entries(runs, cfg, use_session):
    """(array, label, color, linestyle) per run x model, Main-LLM dropped."""
    entries = []
    for i, (label, run_dir) in enumerate(runs):
        ls = RUN_LINESTYLES[i % len(RUN_LINESTYLES)]
        ttft, _, model = C.load_turns(run_dir)
        if use_session:
            data = C.per_session(ttft, model, C.load_sessions(run_dir), sum)
        else:
            data = C.per_request(ttft, model)
        for m, arr in data.items():
            name = cfg.names.get(m, f"M{m}")
            if name == C.MAIN:
                continue
            color = (cfg.type0_colors[m % len(cfg.type0_colors)] if cfg.run_type == 0
                     else cfg.cmaps[m % len(cfg.cmaps)][i % len(cfg.cmaps[0])])
            entries.append((arr, f"{label} {name}", color, ls))
    return entries


def model_col_legend(ax, **kwargs):
    """One column per model that has series, each headed by a text-only entry."""
    handles, labels = ax.get_legend_handles_labels()
    columns = []
    for name, header in ((C.MAIN, "Main-LLM:"), (C.SUB, "Sub-LLM:")):
        items = [(h, l[: -len(name)].strip())
                 for h, l in zip(handles, labels) if l.endswith(name)]
        if items:
            columns.append((header, items))

    entries = []
    for header, items in columns:
        entries.append((Line2D([], [], linestyle="none"), header))
        entries.extend(items)
    if entries:
        ax.legend(*zip(*entries), ncol=max(len(columns), 1), **kwargs)


def annotate_type3(axes, sess_entries=None):
    """Long-tail callouts for the policy comparison at QPS0.5."""
    ax_req, ax_sess = axes
    ax_req.add_patch(Ellipse((450, 0.95), width=500, height=0.2,
                             fill=False, edgecolor="red", linewidth=3, zorder=10))
    ax_req.text(400, 0.82, "longer per-query tail", color="red",
                fontsize=C.FS - 8, ha="left", va="top", zorder=10)
    # Axes-fraction coords so the rotation is not distorted by the x/y scale gap.
    ax_req.add_patch(Ellipse((0.06, 0.75), width=0.12, height=0.45, angle=-5,
                             transform=ax_req.transAxes,
                             fill=False, edgecolor="purple", linewidth=3, zorder=10))
    ax_req.text(30, 0.85, "faster on average", color="purple", fontsize=C.FS - 8,
                ha="left", va="top", zorder=10, rotation=70)

    # Track the data rather than fixed coordinates, so the callout stays on the
    # convergence point when the numbers change.
    if sess_entries:
        import numpy as np
        p95 = [np.percentile(arr, 95) for arr, *_ in sess_entries]
        x_tail, w_tail = float(np.mean(p95)), max(260.0, float(np.ptp(p95)) * 2.6)
    else:
        x_tail, w_tail = 850.0, 250.0
    ax_sess.add_patch(Ellipse((x_tail, 0.95), width=w_tail, height=0.2,
                              fill=False, edgecolor="red", linewidth=3, zorder=10))
    ax_sess.text(x_tail - 0.75 * w_tail, 0.82, "per-task tail similar", color="red",
                 fontsize=C.FS - 8, ha="left", va="top", zorder=10, rotation=65)
    ax_sess.add_patch(Ellipse((0.3, 0.45), width=0.52, height=0.65, angle=-40,
                              transform=ax_sess.transAxes,
                              fill=False, edgecolor="purple", linewidth=3, zorder=10))
    ax_sess.text(30, 1.05, "faster on average", color="purple", fontsize=C.FS - 8,
                 ha="left", va="top", zorder=10, rotation=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="Miro")
    parser.add_argument("--root", default="simulator_output",
                        help="Directory containing the target folders")
    parser.add_argument("--run-type", type=int, default=0, choices=sorted(RUNS_BY_TYPE))
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    names, cmaps, type0_colors = C.model_style(args.target)
    cfg = argparse.Namespace(run_type=args.run_type, names=names,
                             cmaps=cmaps, type0_colors=type0_colors)
    runs = C.resolve_runs(RUNS_BY_TYPE, args.root, args.target, args.run_type)

    req_entries = build_entries(runs, cfg, use_session=False)
    sess_entries = build_entries(runs, cfg, use_session=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.0)
    C.plot_cdf(axes[0], req_entries, "TTFT per query (s)")
    C.plot_cdf(axes[1], sess_entries, "TTFT per task (s)", show_ylabel=False)
    model_col_legend(axes[0], fontsize=C.FS - 6, loc="lower right", labelspacing=0.2)

    if args.run_type == 3:
        annotate_type3(axes, sess_entries)

    out_path = os.path.join(
        args.outdir, f"ttft_{args.target}_type{args.run_type}_req_vs_session.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    C.print_stats([(f"[{mode}] {label}", arr)
                   for mode, entries in [("request", req_entries), ("session", sess_entries)]
                   for arr, label, _, _ in entries], width=32)


if __name__ == "__main__":
    main()
