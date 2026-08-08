"""
figure2.py — MiroThinker prefill/decode scatter (counterpart of owl/analysis/figure3.py)

Prefill vs decode scatter, split by role:
  left   main model, coloured by turn phase (1 / 2-4 / 5+)
  right  summarizer

Input:
    traces/miro_random.csv   (model_id distinguishes main from summarizer)
Usage:
    python figure2.py
Output:
    outputs/Figure2.png
"""

import csv, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

BASE       = Path(__file__).parent           # mirothinker/analysis
DATA       = BASE.parent                     # mirothinker
MERGED_CSV = DATA / "traces" / "miro_random.csv"
OUT_PATH   = BASE / "outputs" / "Figure2.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Styles ─────────────────────────────────────────────────────────────────────
# Colors / markers match the reference figure
PHASE_STYLE = {
    "turn1":   ("#1a7a1a", "o",  "1"),
    "turn2_4": ("#8fca6e", "^",  "2-4"),
    "turn5p":  ("#004d00", "x",  "5+"),
}
SUM_COLOR, SUM_MARKER = "#3a7fc1", "o"

FS = 30 
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 4, "ytick.labelsize": FS - 4,
    "legend.fontsize": FS - 4,
})
APPLY_FIXED_AXES = True
LOG_MAIN_XLIM = (1000, 32768)   
                                
LOG_XLIM      = (10**2.5, 10**4.8)   # sub-LLM / linear fallback
LOG_YLIM      = (10**1.8, 10**4.1)   # shared by both panels


def save_fig(fig, path):
    if APPLY_FIXED_AXES:
        for i, ax in enumerate(fig.get_axes()):
            if ax.get_xscale() == "log":
                ax.set_xlim(LOG_MAIN_XLIM if i == 0 else LOG_XLIM)
                ax.set_ylim(LOG_YLIM)
    fig.savefig(path, dpi=150, bbox_inches="tight")


MODEL_MAIN       = 1   # fine-tuned Qwen-32B
MODEL_SUMMARIZER = 0   # gpt-4o-mini


def classify_phase(raw_rows: list) -> list:
    """
    raw_rows: dicts with num_prefill_tokens, num_decode_tokens, model_id,
              session_id, request_id.
    Returns (prefill, decode, phase) tuples; phase is the main model's turn
    bucket, or "summarizer".
    """
    # Group by session; within each session order by request_id
    by_session = defaultdict(list)
    for r in raw_rows:
        by_session[r["session_id"]].append(r)
    for v in by_session.values():
        v.sort(key=lambda r: r["request_id"])

    result = []
    for session_rows in by_session.values():
        main_turn = 0   # main-model requests seen so far in this session
        for r in session_rows:
            prefill, decode = r["num_prefill_tokens"], r["num_decode_tokens"]
            if r["model_id"] == MODEL_SUMMARIZER:
                result.append((prefill, decode, "summarizer"))
            elif r["model_id"] == MODEL_MAIN:
                main_turn += 1
                phase = "turn1" if main_turn == 1 else ("turn2_4" if main_turn <= 4 else "turn5p")
                result.append((prefill, decode, phase))
    return result


def load_rows():
    """Read the merged trace; merge.py has already applied every filter."""
    raw = []
    with open(MERGED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                raw.append({
                    "num_prefill_tokens": int(row["num_prefill_tokens"]),
                    "num_decode_tokens":  int(row["num_decode_tokens"]),
                    "model_id":           int(row["model_id"]),
                    "session_id":         row["session_id"],
                    "request_id":         int(row["request_id"]),
                })
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(raw)} requests from {MERGED_CSV.name}")
    return raw


def run():
    rows = classify_phase(load_rows())

    fig, (ax_main, ax_sum) = plt.subplots(1, 2, figsize=(13, 6))

    handles = []
    for phase, (color, marker, label) in PHASE_STYLE.items():
        xs = [r[0] for r in rows if r[2] == phase]
        ys = [r[1] for r in rows if r[2] == phase]
        sc = ax_main.scatter(xs, ys, s=40, alpha=1.0, color=color, marker=marker,
                             linewidths=0.8, label=label)
        handles.append(sc)

    ax_main.set_xscale("log"); ax_main.set_yscale("log")
    if APPLY_FIXED_AXES:
        ax_main.set_xlim(LOG_MAIN_XLIM)
    ax_main.set_xlabel("Input tokens")
    ax_main.set_ylabel("Output tokens")
    ax_main.set_title("Main-LLM")
    ax_main.legend(handles=handles,
                   labels=[h.get_label() for h in handles],
                   title="Turns:",
                   title_fontsize=FS - 4,
                   markerscale=1.5,
                   loc="upper right",
                   ncol=3,
                   borderpad=0.1, labelspacing=0.15, handlelength=0.7,
                   handletextpad=0.25, borderaxespad=0.15, columnspacing=0.5)
    ax_main.grid(True, which="major", linewidth=0.4, alpha=0.6)

    sx = [r[0] for r in rows if r[2] == "summarizer"]
    sy = [r[1] for r in rows if r[2] == "summarizer"]
    ax_sum.scatter(sx, sy, s=40, alpha=1.0, color=SUM_COLOR, marker=SUM_MARKER,
                   linewidths=0.8)
    if sx and sy:
        ax_sum.set_xscale("log"); ax_sum.set_yscale("log")
    ax_sum.set_xlabel("Input tokens")
    ax_sum.set_ylabel("")          # no y-label on right plot
    ax_sum.tick_params(labelleft=False)   # same LOG_YLIM as the left panel
    ax_sum.set_title("Sub-LLM")
    ax_sum.grid(True, which="major", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    save_fig(fig, OUT_PATH)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    run()
