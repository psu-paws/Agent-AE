"""
figure4.py  —  OWL vs MiroThinker

Side-by-side token timeline of two traces answering the same GAIA question.
Reads from both agent trees, so the MiroThinker path reaches up to the repo root.
Left  panel : MiroThinker single-agent trace
Right panel : OWL multi-agent trace

Each row = one LLM request.
Stacked bar (left→right): cached prefill | unseen prefill | decode tokens.
Color by model:
  Model 0 — reasoning  (gpt-oss-120b, Qwen)  : green
  Model 1 — non-reason (gpt-4o, gpt-4o-mini) : blue

Usage:
  python figure4.py
Output:
  outputs/Figure4.png
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.patches import Patch


csv.field_size_limit(sys.maxsize)

BASE      = Path(__file__).resolve().parent       # GAIATrace/owl/analysis
DATA      = BASE.parent                           # GAIATrace/owl
ROOT      = DATA.parent                           # GAIATrace
# Illustrative pair; the annotations below are tuned to these two runs.
OWL_CSV   = DATA / "traces" / "session_traces" / "L1_42_0_0412.csv"
MIRO_CSV  = ROOT / "mirothinker" / "traces" / "session_traces" / "3f57289b-8c60-48be-bd80-01f8099ca449.csv"
OUT_PATH  = BASE / "outputs" / "Figure4.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Verbatim from data/gaia_text103.jsonl (task 3f57289b), ground truth 519.
QUESTION = (
    "Task: \"How many at bats did the Yankee with the most walks "
    "in the 1977 regular season have that same season?\""
)

# Labels for any bar in the Miroflow panel.
# Key = 0-based turn index (top→bottom, counting ALL bars — green and blue).
# Omit an index to leave that bar unlabelled.
MIRO_LABELS: dict[float, str] = {
    14: ("404 error", "red", "left"),
}

OWL_LABELS: dict[float, str] = {
    # to be filled in like MIRO_LABELS
}
# ①②③④⑤⑥⑦⑧⑨⑩

# Background color bands. Key = start turn index (0-based); color runs until next key.
# Leave empty ({}) for no background coloring.
MIRO_BG: dict[int, str] = {
    # Groups (bars 1-indexed from top): 1-3, 4-7, 8-10, 11-13, 14-18, rest.
    0:  "#f0f4e8",
    3:  "#ffffff",
    7:  "#f0f4e8",
    10: "#ffffff",
    13: "#f0f4e8",
    18: "#ffffff",
}
OWL_BG: dict[int, str] = {
    # Groups (bars 1-indexed from top): 1-2, 3-11, 12-17, 18-30, 31-49.
    # Colors unused (fill_bg=False); only the partition lines are drawn.
    0:  "#ffffff",
    2:  "#ffffff",
    11: "#ffffff",
    17: "#ffffff",
    30: "#ffffff",
    49: "#ffffff",
}

# Per-group annotation text for the Miroflow panel, one per MIRO_BG group
# (in sorted-key order). "..." renders as a vertical ellipsis.
MIRO_GROUP_TEXTS = [
    "Search 1977 Yankees",
    "Search player statistics",
    "Search Roy ... (Wikipedia)",
    "Search another website",
    "Search Reggie ...",
    "...",
]

# Per-group annotation text for the OWL panel (one per OWL_BG group).
OWL_GROUP_TEXTS: list[str] = [
    "Draft a plan",
    "Search 1977 Yankees",
    "Search player statistics",
    "Verify: Scrape",
    "Verify: Browse",
    "Draft an answer",
]

# Colors: (model_id, segment) → hex
#   model 0 = reasoning  (greenish)
#   model 1 = non-reason (blueish)
COLORS = {
    (0, "cached"): ("#74c476", ""),    # light green
    (0, "unseen"): ("#006837", ""),    # dark green
    (0, "decode"): ("#31a354", ""),    # medium green
    (1, "cached"): ("#41b6c4", ""),    # light blue
    (1, "unseen"): ("#253494", ""),    # dark blue
    (1, "decode"): ("#2c7fb8", ""),    # medium blue
}


# ── helpers ────────────────────────────────────────────────────────────────────

def model_id_owl(tok_str: str) -> int:
    if tok_str.startswith("[200006, 77944"):
        return 1   # gpt-4o
    if tok_str.startswith("[200006, 17360"):
        return 0   # gpt-oss-120b
    return -1


def parse_tokens(tok_str: str) -> list[int]:
    return list(map(int, tok_str[1:-1].split(", ")))


def block_cache_hits(token_ids: list[int], prefill: int, cache: set) -> int:
    parent_hash = 0
    cached = 0
    for i in range(prefill):
        key = hash((parent_hash, token_ids[i]))
        if key in cache:
            cached += 1
            parent_hash = key
        else:
            break
    return cached


def block_cache_add(token_ids: list[int], cache: set) -> None:
    parent_hash = 0
    for tok in token_ids:
        key = hash((parent_hash, tok))
        cache.add(key)
        parent_hash = key


# ── OWL parser ─────────────────────────────────────────────────────────────────

def parse_owl(path: Path) -> list[dict]:
    """Returns list of {cached, unseen, decode, model_id}; recomputes cache via block-hash."""
    turns = []
    cache = {0: set(), 1: set()}   # full block-hash cache per model, never resets

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = int(row["num_prefill_tokens"])
                d = int(row["num_decode_tokens"])
            except (ValueError, KeyError):
                continue
            if d == 0:
                continue
            tok_str = row.get("tokens", "")
            if not tok_str.startswith("["):
                continue
            mid = model_id_owl(tok_str)
            if mid == -1:
                continue
            cur_ids = parse_tokens(tok_str)
            cached = block_cache_hits(cur_ids, p, cache[mid])
            block_cache_add(cur_ids, cache[mid])
            turns.append({"cached": cached, "unseen": p - cached, "decode": d, "model_id": mid})
    return turns


# ── Miroflow parser ────────────────────────────────────────────────────────────

def parse_miroflow(path: Path) -> list[dict]:
    """
    Single CSV with row_kind column.
      main        → model 0 (Qwen/MiroThinker, reasoning, greenish)
      summarizer  → model 1 (gpt-4o-mini, non-reasoning, blueish)
    Separate prefix caches per model_id (different KV pools).
    Rows are kept in file order (interleaved main+summarizer).
    """
    turns = []
    cache = {0: set(), 1: set()}   # full block-hash cache per model, never resets

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = int(row["num_prefill_tokens"])
                d = int(row["num_decode_tokens"])
            except (ValueError, KeyError):
                continue
            if d == 0:
                continue
            kind = row.get("row_kind", "")
            if kind == "main":
                mid = 0
            elif kind == "summarizer":
                mid = 1
            else:
                continue
            tok_str = row.get("tokens", "")
            if not tok_str.startswith("["):
                continue
            cur_ids = parse_tokens(tok_str)
            cached = block_cache_hits(cur_ids, p, cache[mid])
            block_cache_add(cur_ids, cache[mid])
            turns.append({"cached": cached, "unseen": p - cached, "decode": d, "model_id": mid})

    return turns


# ── drawing ────────────────────────────────────────────────────────────────────

FS    = 10      # universal base font size — adjust to scale all text at once
BAR_H = 0.93   # nearly touching (gap = 1 - BAR_H between rows)
ROW_H = 0.06   # inches per turn row


def _bg_color(bg: dict[int, str], i: int) -> str | None:
    """Return the background color for turn i, or None if not covered."""
    keys = sorted(k for k in bg if k <= i)
    return bg[keys[-1]] if keys else None


def draw_panel(ax, turns: list[dict], title: str, show_ylabel: bool = True,
               labels: dict[int, str] | None = None,
               bg: dict[int, str] | None = None, fill_bg: bool = True,
               group_texts: list[str] | None = None, group_num_start: int = 1,
               title_x: float = 0.5):
    n = len(turns)
    bar_widths = []   # bar_widths[i] = total bar width for turn i (top-to-bottom order)

    for i, t in enumerate(turns):
        y   = n - 1 - i
        mid = t["model_id"]
        c   = t["cached"]
        u   = t["unseen"]
        d   = t["decode"]
        bar_widths.append(c + u + d)

        if bg and fill_bg:
            color = _bg_color(bg, i)
            if color:
                ax.axhspan(y - 0.5, y + 0.5, color=color, zorder=0)

        clr_c, htc_c = COLORS[(mid, "cached")]
        clr_u, htc_u = COLORS[(mid, "unseen")]
        clr_d, htc_d = COLORS[(mid, "decode")]
        if c > 0:
            ax.barh(y, c, height=BAR_H, left=0,
                    color=clr_c, hatch=htc_c or None, linewidth=0)
        ax.barh(y, u, height=BAR_H, left=c,
                color=clr_u, hatch=htc_u or None, linewidth=0)
        ax.barh(y, d, height=BAR_H, left=c + u,
                color=clr_d, hatch=htc_d or None, linewidth=0)

    # draw labels — keys can be int/float; value is str or [(text, color), ...]
    if labels:
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        renderer = ax.figure.canvas.get_renderer()
        ax_px_w = ax.get_window_extent(renderer).width

        for key, label in labels.items():
            if not label:
                continue
            y_label = n - 1 - key

            # 3-tuple (text, color, "left") -> placed next to the bar's end
            # (data coords) instead of right-aligned at the panel edge.
            if isinstance(label, tuple) and len(label) == 3 and label[2] == "left":
                seg_text, seg_color, _ = label
                xpos = bar_widths[int(round(key))] + 1500
                ax.text(xpos, y_label, seg_text, va="center", ha="left",
                        fontsize=FS - 3, color=seg_color,
                        clip_on=False, zorder=4)
                continue

            if isinstance(label, str):
                ax.text(0.99, y_label, label, va="center", ha="right",
                        fontsize=FS - 3, color="black",
                        transform=trans, clip_on=False, zorder=4)
            else:
                # (text, color) tuple or list of (text, color) — render right-to-left
                segments = [label] if isinstance(label, tuple) else label
                x = 0.99
                for seg_text, seg_color in reversed(segments):
                    t = ax.text(x, y_label, seg_text, va="center", ha="right",
                                fontsize=FS - 3, color=seg_color,
                                transform=trans, clip_on=False, zorder=4)
                    x -= t.get_window_extent(renderer).width / ax_px_w

    # white dividers clipped to adjacent bar widths (so background bands show beyond bars)
    for y in range(n + 1):
        # turn index above boundary: i_above = n-1-y; below: i_below = n-y
        w_above = bar_widths[n - 1 - y] if 0 <= n - 1 - y < n else 0
        w_below = bar_widths[n - y]     if 0 <= n - y     < n else 0
        w = max(w_above, w_below)
        if w > 0:
            ax.plot([0, w], [y - 0.5, y - 0.5], color="white", linewidth=0.6, zorder=3)

    # horizontal partition lines at background color group boundaries
    if bg:
        for key in sorted(bg.keys()):
            if 0 < key <= n:
                ax.axhline(y=n - key - 0.5, color="#666666", linewidth=0.9,
                           linestyle="-", zorder=3.5)

    # per-group annotation text on the group's center row -- dark grey
    # italic, right-aligned to the panel's right edge, prefixed with a drawn
    # circled number (numbered continuously via group_num_start). "..." ->
    # vertical ellipsis (unnumbered).
    if bg and group_texts:
        gkeys = sorted(bg.keys())
        gtrans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        renderer = ax.figure.canvas.get_renderer()
        ax_px_w = ax.get_window_extent(renderer).width
        num = group_num_start
        for gi, start in enumerate(gkeys):
            if gi >= len(group_texts) or not group_texts[gi]:
                continue
            end = (gkeys[gi + 1] - 1) if gi + 1 < len(gkeys) else n - 1
            mid = (start + end) // 2
            txt = group_texts[gi]
            if txt == "...":
                gmax = max(bar_widths[start:end + 1])
                ax.text(gmax + 8000, n - 1 - mid, "...", rotation=90,
                        va="center", ha="center", fontsize=FS,
                        color="#404040", style="italic", zorder=4)
                continue
            y = n - 1 - mid
            if gi == 1:          # pull the 2nd group's text down half a bar
                y -= 0.5
            if num == 7:         # nudge group 7 up 1 bar
                y += 1
            if num == 8:         # nudge group 8 up 1 bar
                y += 1
            if num == 10:        # nudge group 10 up 4 bars
                y += 4
            t = ax.text(0.99, y, txt,
                        va="center", ha="right", fontsize=FS - 3,
                        color="#404040", style="italic", zorder=4,
                        transform=gtrans, clip_on=False)
            # circled-number badge just left of the text (drawn as a digit in
            # a circle bbox -- DejaVu Sans lacks the ⑪+ unicode glyphs).
            tw = t.get_window_extent(renderer).width / ax_px_w
            ax.text(0.99 - tw - 0.045, y, str(num),
                    va="center", ha="center", fontsize=FS - 5,
                    color="#404040", zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                              ec="#404040", lw=1.0),
                    transform=gtrans, clip_on=False)
            num += 1

    ax.set_yticks([])
    if show_ylabel:
        ax.set_ylabel("Turns (latest ← oldest)", fontsize=FS-2, rotation=90, labelpad=4)
    ax.set_xlabel("Tokens", fontsize=FS-3)
    ax.set_title(title, fontsize=FS-2, fontweight="bold", pad=0, y=0.99, x=title_x)
    ax.set_ylim(-0.5, n - 0.5)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x)))
    )
    ax.tick_params(axis="x", labelsize=FS-3, pad=1)
    ax.tick_params(axis="y", pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    owl_turns  = parse_owl(OWL_CSV)
    vid_turns  = parse_miroflow(MIRO_CSV)
    print(f"OWL turns: {len(owl_turns)}  Miroflow turns: {len(vid_turns)}")

    n_max = max(len(owl_turns), len(vid_turns))
    fig_h = max(2.5, n_max * ROW_H)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(4.0, fig_h),
        gridspec_kw={"wspace": 0.08},
    )

    draw_panel(ax_l, vid_turns,  "MiroThinker", show_ylabel=True,  labels=MIRO_LABELS, bg=MIRO_BG, fill_bg=False,
               group_texts=MIRO_GROUP_TEXTS, group_num_start=1)
    # OWL numbering continues after the left panel's numbered groups.
    owl_num_start = 1 + sum(1 for t in MIRO_GROUP_TEXTS if t and t != "...")
    draw_panel(ax_r, owl_turns,  "OWL",      show_ylabel=False, labels=OWL_LABELS, bg=OWL_BG, fill_bg=False,
               group_texts=OWL_GROUP_TEXTS, group_num_start=owl_num_start, title_x=0.42)

    legend_handles = [
        Patch(facecolor="none", edgecolor="none", label="Main-LLM:"),
        Patch(facecolor="none", edgecolor="none", label="Sub-LLM:"),
        Patch(color=COLORS[(0, "cached")][0], label="Prefill (Cached)", edgecolor="grey", linewidth=0.3),
        Patch(color=COLORS[(1, "cached")][0], label="Prefill (Cached)", edgecolor="grey", linewidth=0.3),
        Patch(color=COLORS[(0, "unseen")][0], label="Prefill (New)",    edgecolor="grey", linewidth=0.3),
        Patch(color=COLORS[(1, "unseen")][0], label="Prefill (New)",    edgecolor="grey", linewidth=0.3),
        Patch(color=COLORS[(0, "decode")][0], label="Decode",           edgecolor="grey", linewidth=0.3),
        Patch(color=COLORS[(1, "decode")][0], label="Decode",           edgecolor="grey", linewidth=0.3),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=FS-2, frameon=True, bbox_to_anchor=(0.5, -0.07),
               handlelength=0.8, handleheight=0.8, borderpad=0.05, labelspacing=0.05,
               handletextpad=0.05, columnspacing=1.0)

    import textwrap
    q_wrapped = "\n".join(textwrap.wrap(QUESTION, width=58))
    fig.suptitle(q_wrapped, fontsize=FS-2.5, y=0.99, va="top", linespacing=1.35,
                 style="italic")

    fig.subplots_adjust(bottom=0.12)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
