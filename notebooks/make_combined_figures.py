"""
Generate publication-quality figures from the combined result JSONs.
Saves PNGs to results/combined_results/.
Run with:  python make_combined_figures.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
COMBINED_DIR = Path(__file__).parent / "results" / "combined_results"
ASSETS_DIR   = COMBINED_DIR          # save alongside the JSON files
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ── data ──────────────────────────────────────────────────────────────────────
CHUNK_SIZES  = ["512", "1024", "2048", "4000", "8000", "full_page"]
X_LABELS     = ["512", "1K", "2K", "4K", "8K", "Full\npage"]
SYSTEMS      = ["Simple RAG", "HtmlRAG", "HyperRAG"]

data: dict[str, dict] = {}
for cs in CHUNK_SIZES:
    with open(COMBINED_DIR / f"results_chunk_{cs}_combined.json", encoding="utf-8") as f:
        data[cs] = json.load(f)

def get_metric(metric: str) -> dict[str, list[float]]:
    """Return {system: [val_per_chunk]} for a given mean_by_system metric key."""
    return {
        sys: [data[cs]["mean_by_system"][sys][metric] for cs in CHUNK_SIZES]
        for sys in SYSTEMS
    }

ret_em  = get_metric("retrieval_em")
sr      = get_metric("supporting_recall")
gen_em  = get_metric("qwen_gen_em")
gen_f1  = get_metric("qwen_gen_f1")

# ── style ─────────────────────────────────────────────────────────────────────
COLORS  = {"Simple RAG": "#2166ac", "HtmlRAG": "#f4a582", "HyperRAG": "#d6604d"}
MARKERS = {"Simple RAG": "o",       "HtmlRAG": "s",       "HyperRAG": "^"}
LINES   = {"Simple RAG": "--",      "HtmlRAG": ":",       "HyperRAG": "-"}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.04,
})

x = np.arange(len(CHUNK_SIZES))

def add_line(ax, vals: dict[str, list], label_map=None):
    for sys in SYSTEMS:
        ax.plot(
            x, vals[sys],
            color=COLORS[sys], marker=MARKERS[sys], linestyle=LINES[sys],
            linewidth=1.5, markersize=5, label=sys,
        )

def fmt_ax(ax, ylabel, ylim=None, title=None):
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, pad=4)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def legend_patches():
    return [
        mpatches.Patch(color=COLORS[s], label=s) for s in SYSTEMS
    ]

# ── Fig A  2×2 combined overview ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)

panels = [
    (axes[0, 0], ret_em,  "Retrieval EM",       (0.1, 1.0),  "Retrieval Exact Match"),
    (axes[0, 1], sr,      "Supporting Recall",  (0.4, 0.9),  "Supporting Recall"),
    (axes[1, 0], gen_em,  "Generation EM",      (0.15, 0.50), "Generation EM"),
    (axes[1, 1], gen_f1,  "Generation F1",      (0.25, 0.55), "Generation F1"),
]
for ax, vals, ylabel, ylim, title in panels:
    add_line(ax, vals)
    fmt_ax(ax, ylabel, ylim, title)
    ax.set_xlabel("Chunk size (chars)")

axes[0, 0].legend(handles=legend_patches(), loc="upper left", framealpha=0.9)
fig.savefig(ASSETS_DIR / "fig_combined_overview.png")
plt.close(fig)
print("Saved fig_combined_overview.png")

# ── Fig B  Supporting Recall (single column, prominent) ─────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
add_line(ax, sr)
fmt_ax(ax, "Supporting Recall", (0.40, 0.86))
ax.set_xlabel("Chunk size (chars)")
ax.set_title("Supporting Recall across chunk sizes")
ax.legend(handles=legend_patches(), loc="upper right", framealpha=0.9)
# annotate HyperRAG peak
peak_i = int(np.argmax(sr["HyperRAG"]))
ax.annotate(
    f"{sr['HyperRAG'][peak_i]:.3f}",
    xy=(peak_i, sr["HyperRAG"][peak_i]),
    xytext=(peak_i - 0.5, sr["HyperRAG"][peak_i] + 0.018),
    fontsize=7, color=COLORS["HyperRAG"],
    arrowprops=dict(arrowstyle="-", color=COLORS["HyperRAG"], lw=0.8),
)
fig.savefig(ASSETS_DIR / "fig_combined_supporting_recall.png")
plt.close(fig)
print("Saved fig_combined_supporting_recall.png")

# ── Fig C  Generation EM + F1 dual (single column) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
add_line(axes[0], gen_em)
fmt_ax(axes[0], "Generation EM", (0.15, 0.50), "Generation EM")
axes[0].set_xlabel("Chunk size (chars)")

add_line(axes[1], gen_f1)
fmt_ax(axes[1], "Generation F1", (0.25, 0.55), "Generation F1")
axes[1].set_xlabel("Chunk size (chars)")

axes[0].legend(handles=legend_patches(), loc="upper left", framealpha=0.9)
fig.savefig(ASSETS_DIR / "fig_combined_generation.png")
plt.close(fig)
print("Saved fig_combined_generation.png")

# ── Fig D  Retrieval EM vs Generation EM (clustered bars @ each chunk) ────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
bar_w = 0.25
bar_colors = list(COLORS.values())

for col, (vals, ylabel, title) in enumerate([
    (ret_em,  "Retrieval EM",   "Retrieval Exact Match"),
    (gen_em,  "Generation EM",  "Generation EM"),
]):
    ax = axes[col]
    for i, sys in enumerate(SYSTEMS):
        offsets = x + (i - 1) * bar_w
        ax.bar(offsets, vals[sys], bar_w, label=sys,
               color=COLORS[sys], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Chunk size (chars)")
    ax.set_title(title, pad=4)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(handles=legend_patches(), loc="upper left", framealpha=0.9)
fig.savefig(ASSETS_DIR / "fig_combined_ret_vs_gen.png")
plt.close(fig)
print("Saved fig_combined_ret_vs_gen.png")

# ── Fig E  Extraction gap (HyperRAG ret_em - gen_em) ─────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
gap_colors = {
    "Simple RAG": COLORS["Simple RAG"],
    "HtmlRAG":    COLORS["HtmlRAG"],
    "HyperRAG":   COLORS["HyperRAG"],
}
bar_w = 0.25
for i, sys in enumerate(SYSTEMS):
    gap = [r - g for r, g in zip(ret_em[sys], gen_em[sys])]
    ax.bar(x + (i - 1) * bar_w, gap, bar_w,
           label=sys, color=COLORS[sys], alpha=0.85, edgecolor="white", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(X_LABELS)
ax.set_ylabel("Retrieval EM − Generation EM")
ax.set_xlabel("Chunk size (chars)")
ax.set_title("Extraction Gap (retrieval EM − generation EM)")
ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(handles=legend_patches(), loc="upper right", framealpha=0.9, ncol=1)
fig.savefig(ASSETS_DIR / "fig_combined_extraction_gap.png")
plt.close(fig)
print("Saved fig_combined_extraction_gap.png")

# ── Fig F  Radar / spider chart at 8K (best operational chunk) ───────────────
labels_radar = ["Ret EM", "Sup Recall", "Gen EM", "Gen F1"]
chunk_8k = "8000"
radar_vals = {
    sys: [
        data[chunk_8k]["mean_by_system"][sys]["retrieval_em"],
        data[chunk_8k]["mean_by_system"][sys]["supporting_recall"],
        data[chunk_8k]["mean_by_system"][sys]["qwen_gen_em"],
        data[chunk_8k]["mean_by_system"][sys]["qwen_gen_f1"],
    ]
    for sys in SYSTEMS
}
N = len(labels_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(3.4, 3.0), subplot_kw=dict(polar=True),
                       constrained_layout=True)
for sys in SYSTEMS:
    vals_r = radar_vals[sys] + radar_vals[sys][:1]
    ax.plot(angles, vals_r, color=COLORS[sys], linewidth=1.8,
            linestyle=LINES[sys], marker=MARKERS[sys], markersize=5, label=sys)
    ax.fill(angles, vals_r, color=COLORS[sys], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, size=8)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], size=6)
ax.grid(True, linestyle=":", linewidth=0.6)
ax.set_title("System comparison at chunk 8K", pad=14, size=9)
ax.legend(handles=legend_patches(), loc="lower right",
          bbox_to_anchor=(1.35, -0.08), framealpha=0.9, fontsize=7)
fig.savefig(ASSETS_DIR / "fig_combined_radar_8k.png")
plt.close(fig)
print("Saved fig_combined_radar_8k.png")

print(f"\nAll figures written to:\n  {COMBINED_DIR}")
