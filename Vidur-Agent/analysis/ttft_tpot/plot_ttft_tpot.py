"""
Per-request (or per-session) TTFT and TPOT distributions.

  --session-level off (default)
      TTFT = prefill_e2e_time, TPOT = normalized decode time, one point per turn.
  --session-level on
      TTFT summed and TPOT averaged over the turns of each session.

Zero-decode turns are excluded from TPOT in both modes.
Layout: 1x2 -- left TTFT CDF, right TPOT CDF. See _common.py for run naming.

Usage:
  python analysis/ttft_tpot/plot_ttft_tpot.py --target Miro --run-type 0
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import _common as C

RUN_LINESTYLES = [":", "-", "--", "-."]

PLACEMENT_A = ["4141-4141", "4141-2321", "2181-4121", "4121-8121"]
PLACEMENT_B = ["4141-4141", "2321-4141", "8121-2141", "4121-2181"]

RUNS_BY_TYPE = {
    0: [("NoPrefix", "{target}/4141-4141-NoPC"),
        ("Prefix",   "{target}/4141-4141-QPS01")],
    1: C.at_qps([("FCFS", "4141-4141"), ("SFCFS", "4141-4141-SFCFS"),
                 (C.SJF180, "4141-4141-SJF180"), (C.SJF, "4141-4141-SJF")], "QPS01"),
    2: C.at_qps([("FCFS", "4141-4141"), (C.SJF, "4141-4141-SJF")], "QPS005"),
    3: C.at_qps([("FCFS", "4141-4141"), ("SFCFS", "4141-4141-SFCFS"),
                 (C.SJF180, "4141-4141-SJF180"), (C.SJF, "4141-4141-SJF")], "QPS05"),
    4: C.placements(PLACEMENT_A, "QPS01"),
    5: C.placements(PLACEMENT_A, "QPS005"),
    6: C.placements(PLACEMENT_A, "QPS05"),
    7: C.placements(PLACEMENT_B, "QPS01"),
    8: C.placements(PLACEMENT_B, "QPS005"),
    9: C.placements(PLACEMENT_B, "QPS05"),
    10: [("QPS0.05", "{target}/4141-4141-QPS005"),
         ("QPS0.5",  "{target}/4141-4141-QPS05")],
    11: [(f"{C.PLACEMENT_LABELS['4141-4141']},QPS0.05", "{target}/4141-4141-QPS005"),
         (f"{C.PLACEMENT_LABELS['2181-4121']},QPS0.05", "{target}/2181-4121-QPS005"),
         (f"{C.PLACEMENT_LABELS['4141-4141']},QPS0.5",  "{target}/4141-4141-QPS05"),
         (f"{C.PLACEMENT_LABELS['4141-2321']},QPS0.5",  "{target}/4141-2321-QPS05")],
}


def build_entries(runs, cfg):
    """(ttft, tpot, label, color, linestyle) per run x model."""
    entries = []
    for i, (label, run_dir) in enumerate(runs):
        ls = RUN_LINESTYLES[i % len(RUN_LINESTYLES)]
        ttft, tpot, model = C.load_turns(run_dir)
        if cfg.session_level:
            sessions = C.load_sessions(run_dir)
            ttft_by_model = C.per_session(ttft, model, sessions, sum)
            tpot_by_model = C.per_session(tpot, model, sessions, np.mean)
        else:
            ttft_by_model = C.per_request(ttft, model)
            tpot_by_model = C.per_request(tpot, model)
        for m in sorted(set(ttft_by_model) | set(tpot_by_model)):
            name = cfg.names.get(m, f"M{m}")
            color = (cfg.type0_colors[m % len(cfg.type0_colors)] if cfg.run_type == 0
                     else cfg.cmaps[m % len(cfg.cmaps)][i % len(cfg.cmaps[0])])
            entries.append((ttft_by_model[m], tpot_by_model[m],
                            f"{label} {name}", color, ls))
    return entries


def row_major_legend(ax, ncol, **kwargs):
    """matplotlib fills legend columns top-to-bottom; reorder so it reads across."""
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)
    nrows = math.ceil(n / ncol)
    slots = [None] * (nrows * ncol)
    for i, (h, l) in enumerate(zip(handles, labels)):
        slots[(i % ncol) * nrows + (i // ncol)] = (h, l)
    entries = [s for s in slots if s is not None]
    ax.legend(*zip(*entries), ncol=ncol, **kwargs)


def type0_legend(ax, cfg):
    """A text header per model column, then the cache on/off runs beneath it."""
    handles, labels = ax.get_legend_handles_labels()
    hmap = dict(zip(labels, handles))
    blank = lambda: Line2D([], [], linestyle="none")
    entries = [
        (blank(), "Main-LLM:"),
        (hmap[f"NoPrefix {C.MAIN}"], "w/o cache"), (hmap[f"Prefix {C.MAIN}"], "w/ cache"),
        (blank(), "Sub-LLM:"),
        (hmap[f"NoPrefix {C.SUB}"], "w/o cache"), (hmap[f"Prefix {C.SUB}"], "w/ cache"),
    ]
    ax.get_figure().legend(
        *zip(*entries), ncol=2, fontsize=C.FS - 4, loc="center",
        bbox_to_anchor=(0.49 if cfg.target.startswith("Owl") else 0.48, 0.5),
        framealpha=1.0, handlelength=1.0, handletextpad=0.25, columnspacing=0.6,
        labelspacing=0.3, borderpad=0.15)


def _arrow(ax, x0, x1, y, color, text=None, fontsize_delta=4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=15,
                arrowprops=dict(color=color, lw=4, arrowstyle="-|>", mutation_scale=22))
    if text:
        ax.text((x0 + x1) / 2, y + 0.07, text, color=color,
                fontsize=C.FS - fontsize_delta, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def annotate_type0(axes, entries, cfg):
    """Show which direction prefix caching moves each model."""
    ax_ttft, ax_tpot = axes
    ttft_by_label = {label: t for t, _, label, _, _ in entries}
    tpot_by_label = {label: p for _, p, label, _, _ in entries}
    pct = lambda table, label, q: float(np.percentile(table[label], q))

    span = ax_ttft.get_xlim()[1]
    for model, q, y in ((C.SUB, 75, 0.75), (C.MAIN, 45, 0.45)):
        x0 = pct(ttft_by_label, f"NoPrefix {model}", q)
        x1 = pct(ttft_by_label, f"Prefix {model}", q)
        if abs(x1 - x0) < 0.02 * span:
            continue                       # too small to be worth an arrow
        faster = x1 < x0
        d = 0.5 if x1 > x0 else -0.5       # shrink both ends toward the middle
        _arrow(ax_ttft, x0 + d, x1 - d, y,
               "purple" if faster else "red",
               "faster" if faster else "slower")

    if cfg.target.startswith("Owl"):
        return

    main_no, main_yes = f"NoPrefix {C.MAIN}", f"Prefix {C.MAIN}"
    sub_no, sub_yes = f"NoPrefix {C.SUB}", f"Prefix {C.SUB}"
    t0, t1 = pct(tpot_by_label, main_no, 50), pct(tpot_by_label, main_yes, 50)
    if t1 - t0 >= t1 * 0.01:
        _arrow(ax_tpot, t0, t1, 0.5, "red", "slower", fontsize_delta=8)

    t0, t1 = pct(tpot_by_label, sub_no, 50), pct(tpot_by_label, sub_yes, 50)
    if t1 - t0 >= t1 * 0.05:
        _arrow(ax_tpot, t0, t1, 0.5, "red")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="Miro")
    parser.add_argument("--root", default="simulator_output",
                        help="Directory containing the target folders")
    parser.add_argument("--run-type", type=int, default=0, choices=sorted(RUNS_BY_TYPE))
    parser.add_argument("--session-level", action="store_true")
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    names, cmaps, type0_colors = C.model_style(args.target)
    cfg = argparse.Namespace(target=args.target, run_type=args.run_type,
                             session_level=args.session_level, names=names,
                             cmaps=cmaps, type0_colors=type0_colors)
    runs = C.resolve_runs(RUNS_BY_TYPE, args.root, args.target, args.run_type)
    entries = build_entries(runs, cfg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.0)
    C.plot_cdf(axes[0], [(t, l, c, s) for t, _, l, c, s in entries],
               "TTFT (s)", extend_left=True)
    C.plot_cdf(axes[1], [(p, l, c, s) for _, p, l, c, s in entries],
               "TPOT (s/token)", show_ylabel=False, extend_left=True)

    if args.run_type == 0:
        # Widen the TPOT panel leftward so the curves clear the legend.
        lo, hi = axes[1].get_xlim()
        axes[1].set_xlim(lo - 0.45 * (hi - lo), hi)
        type0_legend(axes[0], cfg)
        annotate_type0(axes, entries, cfg)
    else:
        row_major_legend(axes[0], ncol=2, fontsize=C.FS - 8, loc="lower right")

    scope = "session" if args.session_level else "request"
    out_path = os.path.join(
        args.outdir, f"ttft_tpot_{args.target}_type{args.run_type}_{scope}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    C.print_stats([(f"{label} {metric}", arr)
                   for ttft, tpot, label, _, _ in entries
                   for metric, arr in (("TTFT", ttft), ("TPOT", tpot))])


if __name__ == "__main__":
    main()
