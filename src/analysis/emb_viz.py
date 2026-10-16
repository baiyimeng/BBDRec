#!/usr/bin/env python3
"""
t-SNE Embedding Visualization (Auto Limits)
=============================================
Plot limits are auto-computed from data distribution.
BBDRec is the reference model (multiplier = 1×).
Each plot shows the relative zoom factor in the top-right corner.

Output: ../imgs/tsne/{dataset}/tSNE_map_{model}_{dataset}.svg
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

# Reuse embedding loading from emb_metrics
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SRC_DIR, "analysis"))
from emb_metrics import load_item_embedding

# GPU device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Configuration
# ============================================================
MODELS = ["sasrec", "bbdrec", "dreamrec", "diffurec"]
# DATASETS = ["baby", "beauty", "ml-100k", "toys"]
DATASETS = ["yelp"]

OUTPUT_DIR = os.path.join(SRC_DIR, "imgs", "emb")
TSNE_RANDOM_STATE = 42

# Auto-limit parameters
AUTO_MARGIN = 1.1  # expand limit by this factor (10% padding)
ROUND_STEP = 1  # round plot limit to nearest step


# ============================================================
# Auto plot-limit helpers
# ============================================================
NICE_MULTS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]


def _round_up(val: float, step: float = ROUND_STEP) -> float:
    """Round up to nearest step (e.g. 37 -> 40, 42 -> 45)."""
    return float(np.ceil(val / step) * step)


def data_extent(coords: np.ndarray) -> float:
    """Maximum absolute coordinate value (radius of bounding box)."""
    return float(max(abs(coords.min()), abs(coords.max())))


def _nice_mult(ratio: float) -> str:
    """Snap ratio to nearest nice multiplier, return label like '5x'."""
    best = min(NICE_MULTS, key=lambda m: abs(m - ratio))
    if best == 1.0:
        return "1x"
    return f"{best:g}x"


def auto_plot_params(
    tsne_results: dict,
    cli_l: float | None = None,
    margin: float = AUTO_MARGIN,
):
    """Compute plot limit and multiplier label for each model.

    BBDRec is the reference — its multiplier = 1x.
    Other models get limits scaled proportionally.

    Returns:
        list of (model_name, coords, lim, mult_label)
    """
    if "bbdrec" not in tsne_results:
        return []

    # --- BBDRec reference ---
    extent_ref = data_extent(tsne_results["bbdrec"])
    if cli_l is not None:
        lim_ref = cli_l
    else:
        lim_ref = _round_up(extent_ref * margin)

    # --- per-model ---
    params = []
    for model, coords in tsne_results.items():
        extent_m = data_extent(coords)
        lim_m = _round_up(extent_m * margin)
        ratio = extent_ref / (extent_m + 1e-9)
        mult_label = _nice_mult(ratio)
        params.append((model, coords, lim_m, mult_label))

    return params


# ============================================================
# Embedding helpers
# ============================================================
def norm(matrix: torch.Tensor) -> torch.Tensor:
    """Z-score normalize along columns (GPU)."""
    return (matrix - matrix.mean(dim=0)) / (matrix.std(dim=0) + 1e-9)


@torch.no_grad()
def load_and_norm(model: str, dataset: str) -> np.ndarray:
    """Load item embeddings, GPU-normalize, return CPU numpy for sklearn."""
    emb = load_item_embedding(model, dataset)  # torch.Tensor on GPU
    emb_norm = norm(emb)
    return emb_norm.cpu().numpy()


def compute_tsne(embeddings: np.ndarray) -> np.ndarray:
    """Run t-SNE on normalized embeddings, return 2D coordinates."""
    tsne = TSNE(
        n_components=2,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
    )
    return tsne.fit_transform(embeddings)


# ============================================================
# Plotting
# ============================================================
def draw_one_auto(
    data: np.ndarray,
    model_name: str,
    dataset: str,
    lim: float,
    mult_label: str,
    save: bool = True,
):
    """Draw a single t-SNE map with multiplier label in top-right corner."""
    sns.set_context("paper")
    sns.set_style("dark")
    plt.subplots(figsize=(5, 5))
    sns.despine(left=True, bottom=True)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    cbar = sns.color_palette("mako", as_cmap=True)
    sns.histplot(
        x=data[:, 0],
        y=data[:, 1],
        bins=30,
        pthresh=0.0,
        stat="percent",
        cmap=cbar,
        cbar=False,
    )
    sns.kdeplot(
        x=data[:, 0],
        y=data[:, 1],
        levels=5,
        color=".6",
        linewidths=1,
    )

    # --- Multiplier label top-right ---
    ax = plt.gca()
    ax.text(
        0.98,
        0.96,
        mult_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="none"
        ),
    )

    if save:
        out_dir = os.path.join(OUTPUT_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"tSNE_map_{model_name}_{dataset}.svg")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"    Saved: {out_path}  (l={lim}, {mult_label})")

    plt.show()
    plt.close()


# ============================================================
# Main
# ============================================================
def process_dataset(dataset: str, base_l: float | None = None, save: bool = True):
    """Compute t-SNE and draw one auto-limits figure per model."""
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset}")
    print(f"{'=' * 60}")

    tsne_results = {}  # model -> 2D coords

    for model in MODELS:
        print(f"\n  [{model}] Loading...")
        try:
            emb_norm = load_and_norm(model, dataset)
        except (FileNotFoundError, KeyError) as e:
            print(f"    SKIP: {e}")
            continue

        print(f"    Embedding: {emb_norm.shape}")
        print(f"    Running t-SNE (sklearn CPU)...")
        tsne_results[model] = compute_tsne(emb_norm)
        print(
            f"    Done. 2D range: [{tsne_results[model].min():.1f}, "
            f"{tsne_results[model].max():.1f}]"
        )

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not tsne_results:
        print(f"  No embeddings loaded, skipping.")
        return

    # --- Auto-compute limits & multipliers ---
    params = auto_plot_params(tsne_results, cli_l=base_l)
    if not params:
        print(f"  BBDRec not in results — cannot auto-compute. Skipping.")
        return

    print(f"\n  Drawing (auto-limits, BBDRec = 1x)...")
    for model, coords, lim_m, mult_label in params:
        draw_one_auto(
            coords, model, dataset, lim=lim_m, mult_label=mult_label, save=save
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="t-SNE Embedding Visualization (auto limits)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Specific dataset (default: all)"
    )
    parser.add_argument(
        "--l",
        type=float,
        default=None,
        help="Override base plot limit for BBDRec (otherwise auto)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save, only display"
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}. Choices: {DATASETS}")
            continue
        process_dataset(ds, base_l=args.l, save=not args.no_save)

    print(f"\n{'=' * 60}")
    print("  All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
