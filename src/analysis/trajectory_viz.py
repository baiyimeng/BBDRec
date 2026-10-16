#!/usr/bin/env python3
"""
Diffusion Forward Noising Trajectory Visualization
=====================================================
Visualize the forward (noising) trajectory of BBDRec, DiffuRec, DreamRec.

- N random test samples (seed=42), same samples across models
- Target item embedding marked with a star (★)
- Forward noising path drawn as circles (○)
- Same sample = same color (consistent across models)
- One joint t-SNE projection per dataset, displayed in separate model panels
- Forward Markov transitions with independent innovations, not fixed-noise marginal curves

Output: imgs/trajectory/{dataset}/trajectory_{model}_{dataset}.svg
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from models.model import DiffusionRecommender
from utils.data import Data_Test
from trajectory_sampling import sample_bridge_path, sample_ddpm_path, trajectory_rng

# ============================================================
# Configuration
# ============================================================
MODELS = ["bbdrec", "diffurec", "dreamrec"]
DATASETS = ["baby", "beauty", "ml-100k", "sports", "toys", "yelp"]
# DATASETS = ["ml-100k"]
N_SAMPLES = 10
SEED = 42

OUTPUT_DIR = os.path.join(SRC_DIR, "imgs", "trajectory")
TSNE_RANDOM_STATE = 42


# Fixed colormap for 10 samples
SAMPLE_CMAP = plt.cm.tab10


# ============================================================
# Model loading
# ============================================================
ITEM_NUM = {
    "ml-100k": 1008,
    "yelp": 64669,
    "sports": 12301,
    "baby": 4731,
    "toys": 7309,
    "beauty": 6086,
}

# T, m, and the DDPM schedule must be supplied from the actual training runs.
# They are not checkpoint tensors and cannot be recovered from a bare state_dict.
RUN_CONFIGS = {}
SAVED_DIR = os.path.join(SRC_DIR, "saved")


def build_args(dataset: str, model_name: str) -> argparse.Namespace:
    """Build argparse.Namespace from config.yaml with model-specific overrides."""
    config_path = os.path.join(SRC_DIR, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override dataset & model
    config["dataset"] = dataset
    config["model"] = model_name
    config["item_num"] = ITEM_NUM[dataset]
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name in ("diffurec", "dreamrec"):
        config["split_onebyone"] = True
        config["parallel_ag"] = False
        if model_name == "diffurec":
            config["is_causal"] = False

    run_config = RUN_CONFIGS.get(dataset, {}).get(model_name)
    required = {"diffusion_steps", "m"} if model_name == "bbdrec" else {"diffusion_steps", "noise_schedule"}
    if not isinstance(run_config, dict) or not required.issubset(run_config):
        raise ValueError(f"Provide actual run settings for {dataset}/{model_name}: {sorted(required)}")
    config.update(run_config)
    config["dataset"], config["model"] = dataset, model_name
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # Full final weights are loaded below; do not depend on a Stage-1 checkpoint.
    config["pretrained"] = False
    config["freeze_emb"] = False
    if config.get("geodesic", False):
        raise ValueError("This visualization implements Euclidean forward transitions only")
    config["geodesic"] = False

    return argparse.Namespace(**config)


def load_model(model_name: str, dataset: str) -> DiffusionRecommender:
    """Load a trained DiffusionRecommender from checkpoint."""
    args = build_args(dataset, model_name)
    device = torch.device(args.device)

    ckpt_path = os.path.join(SAVED_DIR, model_name, dataset, f"{model_name}_{dataset}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Final checkpoint not found: {ckpt_path}")
    model = DiffusionRecommender(args).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {model_name}/{dataset}  from {ckpt_path}")
    return model


# ============================================================
# Data loading
# ============================================================
def load_test_loader(dataset: str, max_len: int = 50):
    """Load test DataLoader for a dataset."""
    args = argparse.Namespace(
        max_len=max_len,
        batch_size=1,
        split_onebyone=False,
        parallel_ag=False,
    )
    data_path = os.path.join(SRC_DIR, "..", "datasets", "data", dataset, "dataset.pkl")
    data_path = os.path.normpath(data_path)
    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    test_data = Data_Test(raw["train"], raw["val"], raw["test"], args)
    return test_data.get_pytorch_dataloaders()


@torch.no_grad()
def sample_test_users(test_loader, n=N_SAMPLES, seed=SEED):
    """Randomly select n test samples (consistent across models via seed)."""
    all_samples = [(hist, tgt) for hist, tgt in test_loader]
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_samples), size=min(n, len(all_samples)), replace=False)
    return [all_samples[i] for i in indices]


# ============================================================
# Forward noising trajectory recording
# ============================================================
@torch.no_grad()
def forward_track_bbdrec(model, target_emb, item_rep_emb, rng):
    """Use the loaded model's schedule and its encoded history endpoint."""
    return sample_bridge_path(
        target_emb.detach().cpu().numpy(), item_rep_emb.detach().cpu().numpy(),
        model.diffu.b_t, model.diffu.d_t, rng,
    )


@torch.no_grad()
def forward_track_ddpm(model, target_emb, rng):
    """Sample successive forward transitions with fresh per-step innovations."""
    return sample_ddpm_path(target_emb.detach().cpu().numpy(), model.diffu.betas, rng)


@torch.no_grad()
def history_endpoint(model, hist):
    """Mirror BBDRec's history encoding before selecting the last valid state."""
    mask_seq = (hist > 0).float()
    seq_emb = model.embed_dropout(model.item_embedding(hist))
    encoded = model.diffu.ag_encoder(seq_emb, mask_seq)
    valid = (hist[0] > 0).nonzero(as_tuple=True)[0]
    if not len(valid):
        raise ValueError("Cannot encode an empty history")
    return encoded[0, valid[-1]].detach()


FORWARD_TRACKERS = {
    "bbdrec": forward_track_bbdrec,
    "diffurec": forward_track_ddpm,
    "dreamrec": forward_track_ddpm,
}


# ============================================================
# Main per-dataset
# ============================================================
@torch.no_grad()
def process_dataset(dataset: str, save: bool = True):
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset}")
    print(f"{'=' * 60}")

    # Load complete, explicitly configured models before processing test data.
    models = {name: load_model(name, dataset) for name in MODELS}
    history_lengths = {model.args.max_len for model in models.values()}
    if len(history_lengths) != 1:
        raise ValueError("Compared models must use the same maximum history length")
    test_loader = load_test_loader(dataset, max_len=history_lengths.pop())
    samples = sample_test_users(test_loader)
    print(f"  Selected {len(samples)} random test samples (seed={SEED})")

    # ---- Collect all forward trajectories + targets ----
    all_traj = {}  # model -> list of traj arrays per sample  [T+1, D]
    all_targets = {}  # model -> list of target arrays per sample  [D]
    history_endpoints = []

    for mname, model in models.items():
        device = model.args.device
        fwd_track = FORWARD_TRACKERS[mname]
        traj_list = []
        tgt_list = []
        for sample_index, (hist, tgt) in enumerate(samples):
            hist = hist.to(device)
            tgt = tgt.to(device)

            # Get target item embedding (clean, last non-pad position)
            tgt_emb_mat = model.item_embedding(tgt)  # [1, max_len, D]
            tgt_idx = (tgt[0] > 0).nonzero(as_tuple=True)[0][-1]
            target_emb = tgt_emb_mat[0, tgt_idx].detach()  # [D]

            rng = trajectory_rng(SEED, DATASETS.index(dataset), MODELS.index(mname), sample_index)
            if mname == "bbdrec":
                item_rep_emb = history_endpoint(model, hist)
                history_endpoints.append(item_rep_emb.cpu().numpy())
                traj = fwd_track(model, target_emb, item_rep_emb, rng)
            else:
                traj = fwd_track(model, target_emb, rng)

            target_pt = target_emb.cpu().numpy().reshape(-1)  # [D]

            traj_list.append(traj)
            tgt_list.append(target_pt)

        all_traj[mname] = traj_list
        all_targets[mname] = tgt_list

    # ---- Gather all points for joint t-SNE ----
    all_flat = []
    point_meta = []  # (model_name, sample_idx, point_type, step_idx)
    for mname in models:
        for s, (traj, tgt) in enumerate(zip(all_traj[mname], all_targets[mname])):
            for step in range(traj.shape[0]):
                all_flat.append(traj[step].reshape(-1))
                point_meta.append((mname, s, "traj", step))
            # x0 already occurs at step 0; do not add a duplicate to t-SNE.

    all_flat = np.stack(all_flat, axis=0)  # [N, D]
    print(f"  Total points for t-SNE: {all_flat.shape[0]}  (dim={all_flat.shape[1]})")

    # ---- t-SNE ----
    print("  Running t-SNE...")
    tsne = TSNE(
        n_components=2, random_state=TSNE_RANDOM_STATE, init="pca", metric="cosine"
    )
    coords_2d = tsne.fit_transform(all_flat)  # [N, 2]
    if save:
        # Retain the raw states/projection so the figure can be reproduced later.
        out_dir = os.path.join(OUTPUT_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "trajectory_data.npz"),
                            states=all_flat, coordinates=coords_2d,
                            history_ids=np.stack([h.cpu().numpy() for h, _ in samples]),
                            target_ids=np.stack([t.cpu().numpy() for _, t in samples]),
                            bbdrec_history_endpoints=np.stack(history_endpoints))
        metadata = {"dataset": dataset, "seed": SEED, "sampling": "forward_markov",
                    "point_meta": point_meta, "run_configs": RUN_CONFIGS[dataset],
                    "effective_configs": {name: vars(model.args) for name, model in models.items()},
                    "checkpoint_paths": {name: os.path.join(SAVED_DIR, name, dataset, f"{name}_{dataset}.pth") for name in models},
                    "tsne": {"random_state": TSNE_RANDOM_STATE, "init": "pca", "metric": "cosine"}}
        with open(os.path.join(out_dir, "trajectory_metadata.json"), "w") as stream:
            json.dump(metadata, stream, indent=2)

    # ---- Per-model drawing ----
    for mname in models:
        model_coords = []
        model_labels = []
        model_point_type = []

        for idx, (mn, s, ptype, step) in enumerate(point_meta):
            if mn == mname:
                model_coords.append(coords_2d[idx])
                model_labels.append(s if ptype == "traj" else f"t_{s}")
                model_point_type.append(ptype)
                if ptype == "traj" and step == 0:
                    # Draw the target star at the exact projected x0, not at
                    # a separately embedded copy of the same input vector.
                    model_coords.append(coords_2d[idx])
                    model_labels.append(f"t_{s}")
                    model_point_type.append("target")

        model_coords = np.array(model_coords)
        print(f"  [{mname}] points: {len(model_coords)}")

        draw_trajectory_figure(
            model_coords,
            model_labels,
            model_point_type,
            mname,
            dataset,
            save=save,
        )


# ============================================================
# Drawing
# ============================================================
def draw_trajectory_figure(
    coords_2d,
    labels,
    point_types,
    model_name,
    dataset,
    save=True,
):
    """Draw one figure: trajectories + star targets."""
    traj_mask = np.array([p == "traj" for p in point_types])
    tgt_mask = np.array([p == "target" for p in point_types])
    traj_coords = coords_2d[traj_mask]
    tgt_coords = coords_2d[tgt_mask]
    traj_labels = [labels[i] for i in range(len(labels)) if traj_mask[i]]
    tgt_labels = [labels[i] for i in range(len(labels)) if tgt_mask[i]]

    sns.set_context("paper")
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.despine(left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw trajectories per sample
    unique_samples = sorted(
        set(l for l in traj_labels if isinstance(l, (int, np.integer)))
    )
    for s in unique_samples:
        sidx = [
            i
            for i, lb in enumerate(traj_labels)
            if isinstance(lb, (int, np.integer)) and lb == s
        ]
        line = traj_coords[sidx]
        color = SAMPLE_CMAP(s % 10)
        # Connect trajectory points with a line
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=3.0, alpha=0.8)
        # All trajectory points as circles (○)
        ax.scatter(
            line[:, 0],
            line[:, 1],
            color=color,
            s=50,
            marker="o",
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        # Dashed line from trajectory start (clean target) -> target star
        tgt_label = f"t_{s}"
        for ti in range(len(tgt_coords)):
            if tgt_labels[ti] == tgt_label:
                ax.plot(
                    [line[0, 0], tgt_coords[ti, 0]],
                    [line[0, 1], tgt_coords[ti, 1]],
                    color=color,
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.5,
                    zorder=2,
                )
                break

    # Draw target stars
    for i in range(len(tgt_coords)):
        lb = tgt_labels[i]
        if isinstance(lb, str) and lb.startswith("t_"):
            s = int(lb.split("_")[1])
            color = SAMPLE_CMAP(s % 10)
            ax.scatter(
                tgt_coords[i, 0],
                tgt_coords[i, 1],
                color=color,
                s=500,
                marker="*",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

    plt.tight_layout()

    if save:
        out_dir = os.path.join(OUTPUT_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"trajectory_{model_name}_{dataset}.svg")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"    Saved: {out_path}")

    plt.show()
    plt.close()


# ============================================================
# CLI
# ============================================================
def main():
    global RUN_CONFIGS, SAVED_DIR
    parser = argparse.ArgumentParser(
        description="Diffusion Forward Markov Trajectory Visualization"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Specific dataset (default: all)"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save, only display"
    )
    parser.add_argument("--run-configs", required=True,
                        help="YAML mapping dataset -> model -> actual training settings")
    parser.add_argument("--saved-dir", default=SAVED_DIR, help="Root of final model checkpoints")
    args = parser.parse_args()
    with open(args.run_configs) as stream:
        RUN_CONFIGS = yaml.safe_load(stream)
    if not isinstance(RUN_CONFIGS, dict):
        raise ValueError("Run configurations must be a YAML mapping")
    SAVED_DIR = os.path.abspath(args.saved_dir)

    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}. Choices: {DATASETS}")
            continue
        process_dataset(ds, save=not args.no_save)

    print(f"\n{'=' * 60}")
    print("  All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
