"""
figure5.py  —  1x3 token timeline for three hand-picked traces
Bar breakdown per request: cached prefill | new prefill | decode.

Input:
    traces/session_traces/{stem}.csv for the stems in STEMS
Usage:
    python figure5.py
Output:
    outputs/Figure5.png
"""

import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

csv.field_size_limit(sys.maxsize)

BASE       = Path(__file__).parent           # GAIATrace/owl/analysis
DATA       = BASE.parent                     # GAIATrace/owl
TRACES_DIR = DATA / "traces" / "session_traces"
OUT_PATH   = BASE / "outputs" / "Figure5.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
FS   = 30   # base font size

STEMS  = ["L1_2_0_0409", "L1_4_0_0410", "L2_80_3_0404"]
TITLES = ["Coding", "Iterative browse", "Web scrape\n& Doc read"]

# Main = gpt-oss-120b,  Sub = gpt-4o
COLORS = {
    (0, "cached"): "#74c476",   # Main cached prefill — light green
    (0, "unseen"): "#006837",   # Main unseen prefill — dark green
    (0, "decode"): "#31a354",   # Main decode         — medium green
    (1, "cached"): "#41b6c4",   # Sub  cached prefill — light teal
    (1, "unseen"): "#253494",   # Sub  unseen prefill — dark navy
    (1, "decode"): "#2c7fb8",   # Sub  decode         — medium blue
}


def model_id(tok):
    if tok.startswith("[200006, 77944"): return 1
    if tok.startswith("[200006, 17360"): return 0
    return -1

BLOCK_SIZE = 1


def session_cache_hits(token_ids, prefill, block_cache, block_size):
    """Cached prefill tokens; stops at the first miss."""
    num_cached = 0
    parent_hash = 0
    for start in range(0, prefill, block_size):
        end = start + block_size
        if end > prefill:
            break
        key = hash((parent_hash, tuple(token_ids[start:end])))
        if key not in block_cache:
            break
        num_cached += block_size
        parent_hash = key
    return num_cached


def session_cache_add(token_ids, block_cache, block_size):
    parent_hash = 0
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        key = hash((parent_hash, tuple(token_ids[start:start + block_size])))
        block_cache.add(key)
        parent_hash = key


def load_turns(stem):
    """Per request: (prefill, decode, model, cached prefill, new prefill)."""
    turns = []
    cache = {0: set(), 1: set()}      # per model, reset each session
    with open(TRACES_DIR / f"{stem}.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p, d = int(row["num_prefill_tokens"]), int(row["num_decode_tokens"])
            except (ValueError, KeyError):
                continue
            if d == 0:
                continue
            tok = row.get("tokens", "")
            mid = model_id(tok)
            if mid == -1:
                continue
            token_ids = list(map(int, tok[1:-1].split(", ")))
            cached = session_cache_hits(token_ids, p, cache[mid], BLOCK_SIZE)
            session_cache_add(token_ids, cache[mid], BLOCK_SIZE)
            turns.append((p, d, mid, cached, p - cached))
    return turns


# ── Load ───────────────────────────────────────────────────────────────────────
all_turns = {s: load_turns(s) for s in STEMS}
ymax      = max(p + d for turns in all_turns.values() for p, d, *_ in turns)

# ── Draw ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)

for idx, (ax, stem, title) in enumerate(zip(axes, STEMS, TITLES)):
    for x, (p, d, mid, cached, unseen) in enumerate(all_turns[stem]):
        if cached > 0:
            ax.bar(x, cached, width=0.8, color=COLORS[(mid, "cached")], linewidth=0)
            ax.bar(x, unseen, width=0.8, bottom=cached, color=COLORS[(mid, "unseen")], linewidth=0)
        else:
            ax.bar(x, p, width=0.8, color=COLORS[(mid, "unseen")], linewidth=0)
        ax.bar(x, d, width=0.8, bottom=p, color=COLORS[(mid, "decode")], linewidth=0)

    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=FS, y=0.93, va="top",
                 bbox=dict(boxstyle="square,pad=0.15", fc="white",
                           ec="none", alpha=0.5))
    if idx == 1:
        ax.set_xlabel("Queries (chronological order)", fontsize=FS - 2)
    ax.tick_params(bottom=False, labelbottom=False, labelsize=FS - 3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1000)}K" if v >= 1000 else str(int(v))))
    ax.spines[:].set_linewidth(0.4)

axes[0].set_ylabel("Tokens", fontsize=FS - 2)

legend_handles = [
    Patch(facecolor="none", edgecolor="none", label="Main-LLM:"),
    Patch(facecolor="none", edgecolor="none", label="Sub-LLM:"),
    Patch(color=COLORS[(0, "cached")], label="Prefill (Cached)", edgecolor="grey", linewidth=0.3),
    Patch(color=COLORS[(1, "cached")], label="Prefill (Cached)", edgecolor="grey", linewidth=0.3),
    Patch(color=COLORS[(0, "unseen")], label="Prefill (New)",    edgecolor="grey", linewidth=0.3),
    Patch(color=COLORS[(1, "unseen")], label="Prefill (New)",    edgecolor="grey", linewidth=0.3),
    Patch(color=COLORS[(0, "decode")], label="Decode",           edgecolor="grey", linewidth=0.3),
    Patch(color=COLORS[(1, "decode")], label="Decode",           edgecolor="grey", linewidth=0.3),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=FS - 2, frameon=True, bbox_to_anchor=(0.5, -0.33),
           handlelength=1.0, handleheight=0.8, borderpad=0.15, labelspacing=0.15,
           handletextpad=0.4, columnspacing=1.0)

fig.tight_layout()
out = OUT_PATH
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
