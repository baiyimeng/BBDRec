import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Data
# =========================

datasets = ["Baby", "Beauty", "ML-100K", "Sports", "Toys", "Yelp"]

methods = [
    "DreamRec",
    "DreamRec+",
    "DiffuRec",
    "DiffuRec+",
    "BBDRec",
]

data = {
    "Baby": {
        "HR": {
            "DreamRec": 0.7648,
            "DreamRec+": 2.9477,
            "DiffuRec": 5.3820,
            "DiffuRec+": 6.4947,
            "BBDRec": 7.4841,
        },
        "NDCG": {
            "DreamRec": 0.2777,
            "DreamRec+": 1.1317,
            "DiffuRec": 2.5649,
            "DiffuRec+": 2.5982,
            "BBDRec": 3.2566,
        },
    },
    "Beauty": {
        "HR": {
            "DreamRec": 0.6815,
            "DreamRec+": 1.5667,
            "DiffuRec": 13.9102,
            "DiffuRec+": 14.9569,
            "BBDRec": 17.1476,
        },
        "NDCG": {
            "DreamRec": 0.2728,
            "DreamRec+": 0.5908,
            "DiffuRec": 7.3326,
            "DiffuRec+": 7.1147,
            "BBDRec": 8.2508,
        },
    },
    "ML-100K": {
        "HR": {
            "DreamRec": 3.4395,
            "DreamRec+": 9.2246,
            "DiffuRec": 16.0688,
            "DiffuRec+": 22.0501,
            "BBDRec": 22.9676,
        },
        "NDCG": {
            "DreamRec": 1.3339,
            "DreamRec+": 3.5123,
            "DiffuRec": 6.5550,
            "DiffuRec+": 8.0732,
            "BBDRec": 8.4818,
        },
    },
    "Sports": {
        "HR": {
            "DreamRec": 0.7432,
            "DreamRec+": 1.1068,
            "DiffuRec": 6.2174,
            "DiffuRec+": 5.7615,
            "BBDRec": 7.9144,
        },
        "NDCG": {
            "DreamRec": 0.2148,
            "DreamRec+": 0.5465,
            "DiffuRec": 3.2067,
            "DiffuRec+": 2.4112,
            "BBDRec": 3.3872,
        },
    },
    "Toys": {
        "HR": {
            "DreamRec": 0.4755,
            "DreamRec+": 1.7918,
            "DiffuRec": 9.8590,
            "DiffuRec+": 9.9694,
            "BBDRec": 12.1264,
        },
        "NDCG": {
            "DreamRec": 0.1731,
            "DreamRec+": 0.6769,
            "DiffuRec": 5.8508,
            "DiffuRec+": 5.0427,
            "BBDRec": 5.8464,
        },
    },
    "Yelp": {
        "HR": {
            "DreamRec": 0.7681,
            "DreamRec+": 0.0878,
            "DiffuRec": 6.7315,
            "DiffuRec+": 6.6610,
            "BBDRec": 7.5349,
        },
        "NDCG": {
            "DreamRec": 0.2477,
            "DreamRec+": 0.0272,
            "DiffuRec": 2.5951,
            "DiffuRec+": 2.6400,
            "BBDRec": 3.0242,
        },
    },
}

# =========================
# 2. ACM style
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
# 3. Plot
# =========================

fig, axes = plt.subplots(2, 6, figsize=(3.8, 1.95), sharey="row")

# Nature / NPG palette.
# Paired methods share a hue family with light/dark variants:
#   DreamRec  / DreamRec+ : light teal  / NPG teal
#   DiffuRec  / DiffuRec+ : NPG sky blue / NPG deep blue
# BBDRec uses the NPG signature red as the hero color for "ours".
colors = [
    "#91D1C2",  # DreamRec   – light teal
    "#00A087",  # DreamRec+  – NPG teal
    "#4DBBD5",  # DiffuRec   – NPG sky blue
    "#3C5488",  # DiffuRec+  – NPG deep blue
    "#E64B35",  # BBDRec     – NPG red (ours)
]


x = np.arange(len(methods))

for col, dataset in enumerate(datasets):
    hr_vals = np.array([data[dataset]["HR"][method] for method in methods])
    ndcg_vals = np.array([data[dataset]["NDCG"][method] for method in methods])

    ax_hr = axes[0, col]
    ax_ndcg = axes[1, col]

    ax_hr.bar(
        x,
        hr_vals,
        color=colors,
        width=0.72,
    )

    ax_ndcg.bar(
        x,
        ndcg_vals,
        color=colors,
        width=0.72,
    )

    for ax in [ax_hr, ax_ndcg]:
        ax.grid(axis="y", linestyle=":", linewidth=0.35, alpha=0.55)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_hr.set_title(dataset, pad=1.5)

axes[0, 0].set_ylabel("HR@20", labelpad=1)
axes[1, 0].set_ylabel("NDCG@20", labelpad=1)

# y-axis ranges
axes[0, 0].set_ylim(0, 25)
axes[1, 0].set_ylim(0, 9)

for ax in axes[0, :]:
    ax.set_yticks([0, 10, 20])
for ax in axes[1, :]:
    ax.set_yticks([0, 4, 8])

# =========================
# 4. Shared legend
# =========================

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(colors))]

fig.legend(
    handles,
    methods,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.12),
    columnspacing=0.55,
    handlelength=0.75,
    handletextpad=0.25,
)

fig.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.98],
    w_pad=0.25,
    h_pad=0.35,
)

plt.savefig("7_plus.pdf", bbox_inches="tight")
print("Saved: 7_plus.pdf")