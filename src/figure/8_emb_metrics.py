import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =========================
# 1. Hardcoded data from emb_metrics.log
# =========================

datasets_display = ["Baby", "Beauty", "ML-100K", "Sports", "Toys", "Yelp"]
models_display  = ["SASRec", "BBDRec", "DreamRec", "DiffuRec"]
n_datasets = len(datasets_display)
n_models   = len(models_display)

# Each entry: [sasrec, bbdrec, dreamrec, diffurec]
data = {
    "Baby": {
        "sv_mean":    [121.05, 204.56,  79.14,  87.72],
        "sv_var":     [285.01, 1519.19, 32.77,  56.07],
        "isotropy":   [0.0941, 0.0103,  0.5170, 0.2595],
        "kl":         [7.327,  10.3616, 0.8863, 1.4755],
    },
    "Beauty": {
        "sv_mean":    [144.33, 186.98,  88.49,  101.11],
        "sv_var":     [453.30, 1457.39, 33.39,  65.62],
        "isotropy":   [0.0874, 0.0192,  0.5617, 0.3147],
        "kl":         [9.2166, 10.3616, 0.7008, 1.3408],
    },
    "ML-100K": {
        "sv_mean":    [67.27,  92.14,   42.19,  42.62],
        "sv_var":     [96.41,  235.44,  34.76,  33.30],
        "isotropy":   [0.0478, 0.0134,  0.2267, 0.1626],
        "kl":         [10.3616, 10.3616, 4.5636, 4.3559],
    },
    "Sports": {
        "sv_mean":    [211.09, 298.67,  147.32, 178.97],
        "sv_var":     [811.81, 3438.21, 139.97, 352.50],
        "isotropy":   [0.0836, 0.0163,  0.1286, 0.1401],
        "kl":         [7.7449, 10.3616, 1.4045, 3.2932],
    },
    "Toys": {
        "sv_mean":    [183.79, 196.85,  96.06,  118.11],
        "sv_var":     [914.78, 1752.01, 34.68,  108.67],
        "isotropy":   [0.0485, 0.0180,  0.5922, 0.2585],
        "kl":         [10.3616, 10.3616, 0.6057, 1.8078],
    },
    "Yelp": {
        "sv_mean":    [660.85, 689.38,  323.68, 606.89],
        "sv_var":     [14639.97, 20706.43, 5237.20, 15004.58],
        "isotropy":   [0.0240, 0.0126,  0.0000, 0.0148],
        "kl":         [10.3616, 10.3616, 0.0000, 10.3616],
    },
}

# =========================
# 2. ACM single-column style
# =========================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# =========================
# 3. Plot — 4 rows × 1 column (full-width grouped bars per metric)
# =========================

fig, axes = plt.subplots(4, 1, figsize=(3.45, 3.55))

colors = [
    "#8491B4",  # SASRec   – NPG slate     (non-diffusion baseline)
    "#E64B35",  # BBDRec   – NPG red       (ours, matches 7_plus)
    "#00A087",  # DreamRec – NPG teal      (matches DreamRec family in 7_plus)
    "#4DBBD5",  # DiffuRec – NPG sky blue  (matches DiffuRec family in 7_plus)
]

row_keys   = ["sv_mean", "sv_var", "isotropy", "kl"]
row_labels = [
    "SV Mean ($\\uparrow$)",
    "SV Var ($\\uparrow$)",
    "Isotropy ($\\downarrow$)",
    "KL to Gauss ($\\uparrow$)",
]
use_log = [True, True, False, False]  # log-scale for SV metrics

# Build 2D arrays: shape (n_models, n_datasets) per metric
metric_arrays = {}
for key in row_keys:
    arr = np.array([data[ds][key] for ds in datasets_display]).T  # (4, 6)
    metric_arrays[key] = arr

x_centers = np.arange(n_datasets)                         # 0..5
bar_w = 0.18                                              # each bar width
offsets = np.arange(n_models) - (n_models - 1) / 2.0      # [-1.5, -0.5, 0.5, 1.5]
bar_positions = [x_centers + offsets[i] * bar_w for i in range(n_models)]

for row_idx, (key, ylabel) in enumerate(zip(row_keys, row_labels)):
    ax = axes[row_idx]
    arr = metric_arrays[key]  # (4, 6)

    for m in range(n_models):
        ax.bar(
            bar_positions[m], arr[m],
            width=bar_w, color=colors[m],
            edgecolor="none", zorder=3,
        )

    # Log scale for SV metrics
    if use_log[row_idx]:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="y", which="minor", length=0)

    # Styling
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets_display, fontsize=6)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.35, alpha=0.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    ax.set_xlim(-0.6, n_datasets - 0.4)

    # Adaptive y-limit with headroom
    vmin, vmax = arr.min(), arr.max()
    if vmin > 0 and use_log[row_idx]:
        lo = vmin * 0.6
        hi = vmax * 2.5
    elif vmin >= 0:
        margin = (vmax - vmin) * 0.20 if vmax != vmin else 0.1
        lo = -margin * 0.2
        hi = vmax + margin
    else:
        margin = abs(vmax - vmin) * 0.20 if vmax != vmin else 0.1
        lo = vmin - margin * 0.2
        hi = vmax + margin
    ax.set_ylim(lo, hi)

# =========================
# 4. Shared legend (single row at top)
# =========================

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_models)]

fig.legend(
    handles,
    models_display,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.96),
    columnspacing=0.55,
    handlelength=0.75,
    handletextpad=0.25,
)

fig.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.935],
    w_pad=0.0,
    h_pad=0.55,
)

plt.savefig("8_emb_metrics.pdf", bbox_inches="tight")
print("Saved: 8_emb_metrics.pdf")
