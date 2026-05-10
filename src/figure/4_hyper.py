import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Raw results
# =========================

datasets = ["Baby", "Beauty", "ML-100K", "Sports", "Toys", "Yelp"]

data = {
    "Baby": {
        "best": {"T": 2, "m": 4, "eta": 1, "lambda": 0.001},
        "T": {
            "x": [2, 4, 8, 16, 32],
            "HR": [7.4841, 7.1773, 7.1346, 7.1766, 7.3642],
            "NDCG": [3.2566, 3.1714, 3.1182, 3.1097, 3.2174],
        },
        "m": {
            "x": [0.5, 1, 2, 4, 8],
            "HR": [7.4841, 7.4756, 7.4756, 7.4841, 7.4756],
            "NDCG": [3.2598, 3.2549, 3.2566, 3.2566, 3.2572],
        },
        "eta": {
            "x": [0.01, 0.1, 1, 10],
            "HR": [7.1586, 7.2358, 7.4841, 6.6519],
            "NDCG": [3.1676, 3.2267, 3.2566, 2.7258],
        },
        "lambda": {
            "x": [0.001, 0.01, 0.1, 1],
            "HR": [7.4841, 7.4236, 7.5008, 6.3773],
            "NDCG": [3.2566, 3.2215, 3.2432, 2.6650],
        },
    },
    "Beauty": {
        "best": {"T": 8, "m": 0.5, "eta": 1, "lambda": 0.001},
        "T": {
            "x": [2, 4, 8, 16, 32],
            "HR": [17.0098, 16.7274, 17.1476, 16.9126, 16.9802],
            "NDCG": [8.0890, 8.1173, 8.2508, 8.1791, 8.2415],
        },
        "m": {
            "x": [0.5, 1, 2, 4, 8],
            "HR": [17.1476, 17.1476, 17.1383, 17.1383, 17.1383],
            "NDCG": [8.2508, 8.2508, 8.2482, 8.2483, 8.2475],
        },
        "eta": {
            "x": [0.01, 0.1, 1, 10],
            "HR": [16.1601, 16.4357, 17.1476, 16.5101],
            "NDCG": [7.9664, 8.0806, 8.2508, 7.7560],
        },
        "lambda": {
            "x": [0.001, 0.01, 0.1, 1],
            "HR": [17.1476, 17.1036, 17.0267, 15.7374],
            "NDCG": [8.2508, 8.2701, 8.0974, 7.1700],
        },
    },
    "ML-100K": {
        "best": {"T": 4, "m": 0.5, "eta": 10, "lambda": 0.001},
        "T": {
            "x": [2, 4, 8, 16, 32],
            "HR": [22.2248, 22.9676, 20.8173, 22.7722, 21.9910],
            "NDCG": [8.2245, 8.4818, 7.6830, 8.3907, 8.2967],
        },
        "m": {
            "x": [0.5, 1, 2, 4, 8],
            "HR": [22.9676, 22.9676, 22.9676, 22.9676, 22.9676],
            "NDCG": [8.4818, 8.4810, 8.4810, 8.4818, 8.4819],
        },
        "eta": {
            "x": [0.01, 0.1, 1, 10],
            "HR": [19.9957, 19.4088, 19.7995, 22.9676],
            "NDCG": [7.5165, 7.5803, 7.6441, 8.4818],
        },
        "lambda": {
            "x": [0.001, 0.01, 0.1, 1],
            "HR": [22.9676, 22.8699, 22.4004, 16.2238],
            "NDCG": [8.4818, 8.4892, 8.4701, 6.1828],
        },
    },
    "Sports": {
        "best": {"T": 2, "m": 0.5, "eta": 0.01, "lambda": 0.001},
        "T": {
            "x": [2, 4, 8, 16, 32],
            "HR": [7.9144, 6.9797, 7.6797, 7.3346, 7.7382],
            "NDCG": [3.3872, 3.1114, 3.3971, 3.2123, 3.3132],
        },
        "m": {
            "x": [0.5, 1, 2, 4, 8],
            "HR": [7.9144, 7.8971, 7.9014, 7.9057, 7.8927],
            "NDCG": [3.3872, 3.3818, 3.3812, 3.3848, 3.3782],
        },
        "eta": {
            "x": [0.01, 0.1, 1, 10],
            "HR": [7.9144, 7.0764, 7.6679, 6.3147],
            "NDCG": [3.3872, 3.2159, 3.2334, 2.6962],
        },
        "lambda": {
            "x": [0.001, 0.01, 0.1, 1],
            "HR": [7.9144, 7.6297, 7.4811, 6.4415],
            "NDCG": [3.3872, 3.3262, 3.2614, 2.9400],
        },
    },
    "Toys": {
        "best": {"T": 16, "m": 4, "eta": 1, "lambda": 0.001},
        "T": {
            "x": [2, 4, 8, 16, 32],
            "HR": [11.5829, 11.7357, 11.2772, 12.1264, 11.9735],
            "NDCG": [5.8504, 5.8949, 5.7075, 5.8464, 5.9854],
        },
        "m": {
            "x": [0.5, 1, 2, 4, 8],
            "HR": [11.6508, 11.6593, 11.6593, 12.1264, 11.6593],
            "NDCG": [5.8764, 5.8774, 5.8789, 5.8464, 5.8758],
        },
        "eta": {
            "x": [0.01, 0.1, 1, 10],
            "HR": [10.5724, 11.8122, 12.1264, 11.7867],
            "NDCG": [5.5972, 5.8985, 5.8464, 5.2432],
        },
        "lambda": {
            "x": [0.001, 0.01, 0.1, 1],
            "HR": [12.1284, 12.1603, 11.4470, 11.4300],
            "NDCG": [5.8464, 5.8186, 5.8875, 5.7078],
        },
    },
    "Yelp": {
        "best": {"T": 2, "m": 4, "eta": 0.1, "lambda": 0.001},
        "T": {
            "x": [2, 4, 8, 16, 32],
            "HR": [7.5349, 7.4674, 7.2284, 7.2906, 7.2836],
            "NDCG": [3.0242, 2.9949, 2.9004, 2.9357, 2.9296],
        },
        "m": {
            "x": [0.5, 1, 2, 4, 8],
            "HR": [7.4462, 7.4857, 7.3650, 7.5349, 7.4686],
            "NDCG": [2.9932, 2.9999, 2.9464, 3.0242, 2.9818],
        },
        "eta": {
            "x": [0.01, 0.1, 1, 10],
            "HR": [7.0750, 7.5349, 7.3070, 6.1205],
            "NDCG": [2.8080, 3.0242, 2.9525, 2.4947],
        },
        "lambda": {
            "x": [0.001, 0.01, 0.1, 1],
            "HR": [7.5349, 7.4945, 7.2828, 5.7676],
            "NDCG": [3.0242, 2.9831, 2.9104, 2.2706],
        },
    },
}


# =========================
# 2. Utility functions
# =========================


def relative_change_to_best(values):
    """
    Normalize by the best value within each sensitivity group.
    Best point becomes 0.
    """
    values = np.asarray(values, dtype=float)
    best = values.max()
    return (values - best) / best * 100.0


def relative_change_to_first(values):
    """
    Normalize by the first value.
    Used for lambda, where lambda=0.001 is treated as the default/best setting.
    """
    values = np.asarray(values, dtype=float)
    ref = values[0]
    return (values - ref) / ref * 100.0


# =========================
# 3. ACM-like style
# =========================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =========================
# 4. Plot
# =========================

params = [
    ("T", r"$T$"),
    ("m", r"$m$"),
    ("eta", r"$\eta$"),
    ("lambda", r"$\lambda$"),
]

markers = {
    "Baby": "o",
    "Beauty": "s",
    "ML-100K": "^",
    "Sports": "D",
    "Toys": "v",
    "Yelp": "P",
}

colors = {
    "Baby": "#1f77b4",
    "Beauty": "#ff7f0e",
    "ML-100K": "#2ca02c",
    "Sports": "#d62728",
    "Toys": "#9467bd",
    "Yelp": "#8c564b",
}

# 2 rows x 4 columns, still ACM single-column style
fig, axes = plt.subplots(2, 4, figsize=(3.55, 2.35), sharey="row")

for col, (param_key, param_symbol) in enumerate(params):
    ax_hr = axes[0, col]
    ax_ndcg = axes[1, col]

    for dataset in datasets:
        x_labels = data[dataset][param_key]["x"]
        x_pos = np.arange(len(x_labels))

        hr = data[dataset][param_key]["HR"]
        ndcg = data[dataset][param_key]["NDCG"]

        if param_key == "lambda":
            hr_delta = relative_change_to_first(hr)
            ndcg_delta = relative_change_to_first(ndcg)
        else:
            hr_delta = relative_change_to_best(hr)
            ndcg_delta = relative_change_to_best(ndcg)

        ax_hr.plot(
            x_pos,
            hr_delta,
            marker=markers[dataset],
            color=colors[dataset],
            linewidth=0.70,
            markersize=2.4,
            markeredgewidth=0.35,
            label=dataset,
        )

        ax_ndcg.plot(
            x_pos,
            ndcg_delta,
            marker=markers[dataset],
            color=colors[dataset],
            linewidth=0.70,
            markersize=2.4,
            markeredgewidth=0.35,
            label=dataset,
        )

    for ax in [ax_hr, ax_ndcg]:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.85)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.55)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(x) for x in x_labels])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_ndcg.set_xlabel(param_symbol, labelpad=1)

axes[0, 0].set_ylabel(r"$\Delta$HR@20 (%)", labelpad=1)
axes[1, 0].set_ylabel(r"$\Delta$NDCG@20 (%)", labelpad=1)

# Lambda=1 may cause larger degradation, so use a slightly wider y range.
axes[0, 0].set_ylim(-30, 3)
axes[1, 0].set_ylim(-30, 3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 1.035),
    columnspacing=0.75,
    handlelength=0.9,
    handletextpad=0.25,
)

fig.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.92],
    w_pad=0.35,
    h_pad=0.55,
)

plt.savefig("4_hyper.pdf", bbox_inches="tight")
plt.show()
