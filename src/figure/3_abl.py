import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Data
# =========================

datasets = ["Baby", "Beauty", "ML-100K", "Sports", "Toys", "Yelp"]

variants = [
    "w/o pretrain",
    "w/ frozen emb",
    "w/o warmup",
    "w/o MSE",
    "w/o CE",
    "w/ MLP denoiser",
]

data = {
    "Baby": {
        "HR": {
            "Base": 7.4841,
            "w/o pretrain": 4.3702,
            "w/ frozen emb": 4.8341,
            "w/o warmup": 6.2919,
            "w/o MSE": 7.1084,
            "w/o CE": 2.8705,
            "w/ MLP denoiser": 7.3382,
        },
        "NDCG": {
            "Base": 3.2566,
            "w/o pretrain": 2.1321,
            "w/ frozen emb": 1.8216,
            "w/o warmup": 2.8630,
            "w/o MSE": 3.1616,
            "w/o CE": 1.0633,
            "w/ MLP denoiser": 3.1689,
        },
    },
    "Beauty": {
        "HR": {
            "Base": 17.1476,
            "w/o pretrain": 12.7030,
            "w/ frozen emb": 8.8858,
            "w/o warmup": 15.7745,
            "w/o MSE": 15.9344,
            "w/o CE": 2.6786,
            "w/ MLP denoiser": 16.1457,
        },
        "NDCG": {
            "Base": 8.2508,
            "w/o pretrain": 6.8289,
            "w/ frozen emb": 3.3175,
            "w/o warmup": 7.8375,
            "w/o MSE": 7.9077,
            "w/o CE": 0.9803,
            "w/ MLP denoiser": 7.8687,
        },
    },
    "ML-100K": {
        "HR": {
            "Base": 22.9676,
            "w/o pretrain": 15.2472,
            "w/ frozen emb": 17.9243,
            "w/o warmup": 19.9760,
            "w/o MSE": 19.1938,
            "w/o CE": 13.7026,
            "w/ MLP denoiser": 22.7910,
        },
        "NDCG": {
            "Base": 8.4818,
            "w/o pretrain": 5.8156,
            "w/ frozen emb": 7.0224,
            "w/o warmup": 7.3398,
            "w/o MSE": 7.5283,
            "w/o CE": 4.7744,
            "w/ MLP denoiser": 8.6835,
        },
    },
    "Sports": {
        "HR": {
            "Base": 7.9144,
            "w/o pretrain": 5.8016,
            "w/ frozen emb": 3.2162,
            "w/o warmup": 7.7197,
            "w/o MSE": 7.6156,
            "w/o CE": 2.1929,
            "w/ MLP denoiser": 7.0124,
        },
        "NDCG": {
            "Base": 3.3872,
            "w/o pretrain": 3.0323,
            "w/ frozen emb": 1.2785,
            "w/o warmup": 3.5651,
            "w/o MSE": 3.3346,
            "w/o CE": 0.9263,
            "w/ MLP denoiser": 3.2237,
        },
    },
    "Toys": {
        "HR": {
            "Base": 12.1284,
            "w/o pretrain": 9.6977,
            "w/ frozen emb": 4.2035,
            "w/o warmup": 11.5065,
            "w/o MSE": 10.8781,
            "w/o CE": 1.6644,
            "w/ MLP denoiser": 12.8991,
        },
        "NDCG": {
            "Base": 5.8464,
            "w/o pretrain": 5.7430,
            "w/ frozen emb": 1.6228,
            "w/o warmup": 5.8305,
            "w/o MSE": 5.6743,
            "w/o CE": 0.6142,
            "w/ MLP denoiser": 6.4695,
        },
    },
    "Yelp": {
        "HR": {
            "Base": 7.5349,
            "w/o pretrain": 6.4189,
            "w/ frozen emb": 0.2334,
            "w/o warmup": 7.0716,
            "w/o MSE": 6.9188,
            "w/o CE": 1.8390,
            "w/ MLP denoiser": 7.1619,
        },
        "NDCG": {
            "Base": 3.0242,
            "w/o pretrain": 2.5302,
            "w/ frozen emb": 0.0765,
            "w/o warmup": 2.8079,
            "w/o MSE": 2.7752,
            "w/o CE": 0.7424,
            "w/ MLP denoiser": 2.8680,
        },
    },
}

# =========================
# 2. Helper
# =========================


def relative_changes(dataset, metric):
    base = data[dataset][metric]["Base"]
    return np.array(
        [(data[dataset][metric][variant] - base) / base * 100.0 for variant in variants]
    )


# =========================
# 3. ACM style
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
# 4. Plot
# =========================

fig, axes = plt.subplots(2, 6, figsize=(3.9, 1.85), sharey="row")

# NPG palette: keep the most informative variant (w/o CE, the most damaging
# one) in the signature NPG red so the eye is naturally drawn to it.
colors = [
    "#3C5488",  # w/o pretrain    – NPG deep blue
    "#4DBBD5",  # w/ frozen emb   – NPG sky blue
    "#F39B7F",  # w/o warmup      – NPG salmon
    "#00A087",  # w/o MSE         – NPG teal
    "#E64B35",  # w/o CE          – NPG red       (most critical ablation)
    "#8491B4",  # w/ MLP denoiser – NPG slate
]

x = np.arange(len(variants))

for col, dataset in enumerate(datasets):
    hr_delta = relative_changes(dataset, "HR")
    ndcg_delta = relative_changes(dataset, "NDCG")

    ax_hr = axes[0, col]
    ax_ndcg = axes[1, col]

    ax_hr.bar(
        x,
        hr_delta,
        color=colors,
        width=0.72,
    )

    ax_ndcg.bar(
        x,
        ndcg_delta,
        color=colors,
        width=0.72,
    )

    for ax in [ax_hr, ax_ndcg]:
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.35, alpha=0.55)

        # Remove x-axis labels and ticks
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_hr.set_title(dataset, pad=1.5)

axes[0, 0].set_ylabel(r"$\Delta$HR (%)", labelpad=1)
axes[1, 0].set_ylabel(r"$\Delta$NDCG (%)", labelpad=1)

# y-axis ranges
axes[0, 0].set_ylim(-100, 15)
axes[1, 0].set_ylim(-100, 15)

for ax in axes[0, :]:
    ax.set_yticks([-100, -80, -40, 0])
for ax in axes[1, :]:
    ax.set_yticks([-100, -80, -40, 0])

# =========================
# 5. Shared legend in one row
# =========================

legend_labels = [
    "w/o pretrain",
    "w/ frozen emb",
    "w/o warmup",
    "w/o MSE",
    "w/o CE",
    "w/ MLP denoiser",
]

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(colors))]

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 1.12),
    columnspacing=0.35,
    handlelength=0.65,
    handletextpad=0.2,
)

fig.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.98],
    w_pad=0.25,
    h_pad=0.35,
)

plt.savefig("3_abl.pdf", bbox_inches="tight")
print("Saved: 3_abl.pdf")