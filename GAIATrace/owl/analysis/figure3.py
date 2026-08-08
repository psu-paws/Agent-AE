"""
figure3.py  —  prefill/decode scatter, split by agent role

Input:
    traces/session_traces/{stem}.csv for each stem in traces/selected_set.txt
    (the per-trace CSVs are the only place the `agent` column exists)
Usage:
    python figure3.py
Output:
    outputs/Figure3.png
"""

import csv, math, sys
from pathlib import Path

BASE       = Path(__file__).parent           # GAIATrace/owl/analysis
DATA       = BASE.parent                     # GAIATrace/owl
TRACES_DIR = DATA / "traces" / "session_traces"
MANIFEST   = DATA / "traces" / "selected_set.txt"
OUT_PATH   = BASE / "outputs" / "Figure3.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

csv.field_size_limit(sys.maxsize)

AGENT_STYLES = {
    0: ("#2c7fb8", "^", "Plan"),
    1: ("#31a354", "o", "Coord"),
    2: ("#a1dab4", "o", "WebSearch"),
    3: ("#bae4bc", "s", "Code"),
    4: ("#006837", "^", "WebSum"),
    5: ("#78c679", "D", "WebPlan"),
    6: ("#41b6c4", "s", "WebAction"),
    7: ("#253494", "D", "Ans"),
    8: ("#c79fef", "P", "Doc"),
    9: ("#9acd32", "P", "etc"),
}

MODEL_AGENT_GROUPS = {
    "Main-LLM":  [1, 3, 4, 5, 9],
    "Sub-LLM":   [0, 8, 6, 2, 7],
}

# Draw order, bottom to top. Legend order is unaffected.
ZORDER = {3: 1, 5: 2, 1: 3, 4: 4, 9: 5,
          2: 1, 6: 2, 0: 3, 8: 4, 7: 5}

FS = 30
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 4, "ytick.labelsize": FS - 4,
    "legend.fontsize": FS - 4,
})
LOG_XLIM, LOG_YLIM = (1e2, 1e5), (1e1, 2*1e4)


def save_fig(fig, path):
    for ax in fig.get_axes():
        if ax.get_xscale() == "log":
            ax.set_xlim(getattr(ax, "_log_xlim", LOG_XLIM)); ax.set_ylim(LOG_YLIM)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def load_rows():
    """
    (prefill, decode, agent) for every request in the selected set.

    Read straight from session_traces: parser writes the `agent` and merge.py drops it.
    """
    if not MANIFEST.exists():
        raise SystemExit("selected_set.txt not found — run select_set.py first")
    stems = MANIFEST.read_text(encoding="utf-8").split()
    rows = []
    for stem in stems:
        path = TRACES_DIR / f"{stem}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append((int(row["num_prefill_tokens"]),
                                 int(row["num_decode_tokens"]),
                                 int(row["agent"])))
                except (ValueError, KeyError):
                    continue
    return rows


def run(_unused=None):
    rows = load_rows()
    print(f"Loaded {len(rows)} requests from {TRACES_DIR.name}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    for ax, (model_name, agent_ids) in zip(axes, MODEL_AGENT_GROUPS.items()):
        for aid in agent_ids:
            color, marker, label = AGENT_STYLES.get(aid, ("#cccccc", "o", f"Agent {aid}"))
            xs = [r[0] for r in rows if r[2] == aid]
            ys = [r[1] for r in rows if r[2] == aid]
            if not xs:
                continue
            ax.scatter(xs, ys, s=24, alpha=0.95, color=color, marker=marker,
                       linewidths=0.5, label=f"{label}",
                       zorder=ZORDER.get(aid, 1))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(model_name)
        ax.set_xlabel("Input tokens")
        ax.set_ylabel("Output tokens")
        ax.legend(markerscale=2, loc="upper right", ncol=3,
                  borderpad=0.1, labelspacing=0.15, handlelength=0.7,
                  handletextpad=0.25, borderaxespad=0.15, columnspacing=0.5)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.75)

    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1]._log_xlim = (7 * 10**2, 10**5)

    # Hand-placed cluster ellipses on the Main-LLM plot. 
    ax0 = axes[0]
    _xlo, _xhi = LOG_XLIM
    _ylo, _yhi = LOG_YLIM

    def _frac(v, lo, hi):
        return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④"}
    # data_x, data_y, width_frac, height_frac, number, label_x, label_y
    for cx, cy, w, h, num, lx, ly in [
        (2.2*1e2, 2e3,     0.1,  0.2, "4", 0.05, 0.60), 
        (1e4,     1.3*1e2, 0.4,  0.2, "1", 0.65, 0.15), 
        (1e4,     1.5*1e2, 0.16, 0.3, "3", 0.9, 0.3), 
        (1.2*1e3, 1.5*1e2, 0.35, 0.3, "2", 0.24, 0.18), 
    ]:
        fx = _frac(cx, _xlo, _xhi)
        fy = _frac(cy, _ylo, _yhi)
        ax0.add_patch(Ellipse((fx, fy), width=w, height=h,
                              transform=ax0.transAxes, fill=False,
                              edgecolor="red", linewidth=2.5, zorder=20))
        ax0.text(lx, ly, CIRCLED[num], transform=ax0.transAxes,
                 color="red", ha="center", va="center", fontsize=FS, zorder=21)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    out = OUT_PATH
    save_fig(fig, out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    run()