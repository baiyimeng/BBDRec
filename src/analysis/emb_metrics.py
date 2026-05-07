#!/usr/bin/env python3
"""
Embedding Metrics Analysis (GPU-accelerated)
=============================================
Analyze item embedding quality for SASRec, BBDRec, DreamRec, DiffuRec
across all datasets: baby, beauty, ml-100k, sports, toys, yelp.

Metrics computed:
  - Embedding Variance       (average per-dimension variance)
  - Cosine Similarity        (mean pairwise cosine similarity)
  - Singular Value Spectrum  (top-5 singular values)
  - Singular Value Entropy   (entropy of normalized singular values)
  - Singular Value Variance  (variance of singular values)
  - Isotropy Score           (min_eig / max_eig of covariance)
  - KL Divergence to Gaussian
  - Mutual Information       (PCA-1D label-bin mutual info) [CPU]
  - Covariance Matrix Entropy
  - Silhouette Score         (KMeans clustering, n_clusters=16) [CPU]

All heavy linear-algebra operations run on GPU via PyTorch.
Only sklearn/scipy wrappers (KMeans, silhouette_score, mutual_info_score,
scipy.stats.entropy) run on CPU since they lack native GPU support.

Output: prints per-model per-dataset results and saves to CSV.
"""

import csv
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, silhouette_score


# ============================================================
# Configuration
# ============================================================
MODELS = ["sasrec", "bbdrec", "dreamrec", "diffurec"]
DATASETS = ["baby", "beauty", "ml-100k", "sports", "toys", "yelp"]
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR = os.path.join(SRC_DIR, "saved")
OUTPUT_CSV = os.path.join(SRC_DIR, "analysis", "embedding_results.csv")

# GPU device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Device] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ============================================================
# Embedding Loading
# ============================================================
def load_item_embedding(model: str, dataset: str) -> torch.Tensor:
    """
    Load the item_embedding.weight from a saved model checkpoint.
    Returns a torch.Tensor on DEVICE, shape (num_items, hidden_dim), excluding pad token.
    """
    path1 = os.path.join(SAVED_DIR, model, dataset, f"{model}_{dataset}.pth")
    path2 = os.path.join(SAVED_DIR, model, dataset, "pretrain.pth")
    path = path1 if os.path.exists(path1) else path2

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found for {model}/{dataset}.\n"
            f"  Tried: {path1}\n"
            f"  Tried: {path2}"
        )

    print(f"  Loading: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)

    if "item_embedding.weight" in saved:
        weight = saved["item_embedding.weight"]
    elif "model_state_dict" in saved:
        weight = saved["model_state_dict"]["item_embedding.weight"]
    else:
        for key in saved:
            if "embed" in key.lower() and "weight" in key.lower():
                weight = saved[key]
                break
        else:
            raise KeyError(
                f"Cannot find item_embedding weight in {path}. "
                f"Keys: {list(saved.keys())[:10]}"
            )

    # Move to GPU, exclude padding token (index 0)
    emb = weight.detach().to(DEVICE)
    return emb[1:]


# ============================================================
# Metric Functions  (torch on GPU)
# ============================================================
def _to_np(t: torch.Tensor) -> np.ndarray:
    """Helper: detach & move tensor to CPU numpy."""
    return t.detach().cpu().numpy()


def norm(matrix: torch.Tensor) -> torch.Tensor:
    """Z-score normalize along columns."""
    return (matrix - matrix.mean(dim=0)) / (matrix.std(dim=0) + 1e-9)


def compute_embedding_variance(embeddings: torch.Tensor) -> torch.Tensor:
    """Per-dimension variance."""
    return torch.var(embeddings, dim=0)


def compute_cosine_similarity_mean(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity — O(nd) memory, no n×n matrix.

    Uses identity: mean(cos) = (||Σ e_i||² - n) / (n·(n-1))
    where e_i are L2-normalized embedding vectors.
    """
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sum_vec = emb_norm.sum(dim=0)
    sum_sq = (sum_vec ** 2).sum()
    n = embeddings.shape[0]
    return (sum_sq - n) / (n * (n - 1))


def compute_mutual_distance_mean(embeddings: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Mean pairwise Euclidean distance — chunked cdist, O(chunk·n) memory.

    Splits rows into chunks to avoid materializing the full n×n distance matrix.
    Diagonal (self-distance = 0) is excluded from the mean.
    """
    n = embeddings.shape[0]
    total = torch.tensor(0.0, device=embeddings.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]
        dists = torch.cdist(chunk, embeddings, p=2)  # [chunk, n]
        total += dists.sum()
        del dists
    return total / (n * (n - 1))


def compute_mutual_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance matrix — WARNING: allocates n×n GPU memory.

    Only suitable for small item sets (<10k). For large sets use
    compute_mutual_distance_mean() instead.
    """
    sq_norm = (embeddings ** 2).sum(dim=1, keepdim=True)
    dist_sq = sq_norm + sq_norm.T - 2 * (embeddings @ embeddings.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq)


def compute_singular_value_spectrum(embeddings: torch.Tensor) -> torch.Tensor:
    """Singular values of standardized embeddings (GPU SVD)."""
    embeddings_norm = norm(embeddings)
    _, s, _ = torch.linalg.svd(embeddings_norm, full_matrices=False)
    return s


def singular_value_entropy(singular_values: torch.Tensor) -> float:
    """Entropy of normalized singular values."""
    sv_np = _to_np(singular_values)
    normalized = sv_np / (np.sum(sv_np) + 1e-9)
    return float(entropy(normalized))


def singular_value_variance(singular_values: torch.Tensor) -> float:
    """Variance of singular values."""
    return float(torch.var(singular_values).cpu())


def evaluate_singular_value(singular_values: torch.Tensor) -> tuple:
    return singular_value_entropy(singular_values), singular_value_variance(singular_values)


def isotropy_score(embeddings: torch.Tensor) -> float:
    """Isotropy: min(eigenvalue) / max(eigenvalue) of covariance matrix (GPU)."""
    embeddings_norm = norm(embeddings)
    n = embeddings_norm.shape[0]
    cov = (embeddings_norm.T @ embeddings_norm) / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    return float((torch.min(eigvals) / (torch.max(eigvals) + 1e-9)).cpu())


def centered_kernel_alignment(
    X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0
) -> float:
    """Centered Kernel Alignment between two embedding matrices (GPU)."""
    X = norm(X)
    Y = norm(Y)

    # RBF kernel on GPU (uses full n×n matrix — ensure n < 10k or enough GPU memory)
    sq_dists_X = compute_mutual_distance_matrix(X) ** 2
    K_X = torch.exp(-gamma * sq_dists_X)

    sq_dists_Y = compute_mutual_distance_matrix(Y) ** 2
    K_Y = torch.exp(-gamma * sq_dists_Y)

    # Center
    K_X_centered = K_X - K_X.mean(dim=0) - K_X.mean(dim=1, keepdim=True) + K_X.mean()
    K_Y_centered = K_Y - K_Y.mean(dim=0) - K_Y.mean(dim=1, keepdim=True) + K_Y.mean()

    numerator = torch.trace(K_X_centered.T @ K_Y_centered)
    denom = torch.norm(K_X_centered) * torch.norm(K_Y_centered)
    return float((numerator / (denom + 1e-9)).cpu())


def kl_to_gaussian(embeddings: torch.Tensor) -> float:
    """KL divergence from embedding distribution to isotropic Gaussian (GPU)."""
    emb_norm = norm(embeddings)
    n = emb_norm.shape[0]
    d = embeddings.shape[1]
    cov = (emb_norm.T @ emb_norm) / (n - 1)
    trace_cov = torch.trace(cov)
    det_cov = torch.linalg.det(cov)
    kl = 0.5 * (trace_cov - d - torch.log(torch.clamp(det_cov, min=1e-9)))
    return float(kl.cpu())


def mutual_info(embeddings: torch.Tensor) -> float:
    """Mutual information between item ID and PCA-1D bin. Runs on CPU (sklearn)."""
    emb_np = _to_np(embeddings)
    labels = np.arange(emb_np.shape[0])
    pca_proj = PCA(n_components=1).fit_transform(emb_np).squeeze()
    _, bin_edges = np.histogram(pca_proj, bins=10)
    bins = np.digitize(pca_proj, bin_edges[:-1])
    return float(mutual_info_score(labels, bins))


def covariance_entropy(embeddings: torch.Tensor) -> float:
    """Entropy of normalized eigenvalues of the covariance matrix (GPU)."""
    n = embeddings.shape[0]
    cov = (embeddings.T @ embeddings) / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=1e-9)
    eigvals = eigvals / eigvals.sum()
    eigvals_np = _to_np(eigvals)
    return float(entropy(eigvals_np))


def compute_silhouette_score(embeddings: torch.Tensor, n_clusters: int = 16) -> float:
    """Silhouette score using KMeans clustering. Runs on CPU (sklearn)."""
    emb_np = _to_np(embeddings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb_np)
        score = silhouette_score(emb_np, labels)
    return float(score)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_embeddings(embeddings: torch.Tensor) -> dict:
    """Compute all embedding quality metrics. Returns dict of rounded values.

    Wrapped in torch.no_grad() to avoid building computation graph and save GPU memory.
    """
    results = {}

    # Basic metrics (GPU)
    variances = compute_embedding_variance(embeddings)
    results["Embedding Variance"] = round(float(torch.mean(variances).cpu()), 4)

    results["Cosine Similarity"] = round(
        float(compute_cosine_similarity_mean(embeddings).cpu()), 4
    )

    results["Mutual Distance Mean"] = round(
        float(compute_mutual_distance_mean(embeddings).cpu()), 4
    )

    # Singular value spectrum (GPU SVD)
    singular_values = compute_singular_value_spectrum(embeddings)
    results["Top 5 Singular Values"] = str(
        np.round(_to_np(singular_values[:5]), 4)
    )
    sv_entropy, sv_variance = evaluate_singular_value(singular_values)
    results["Singular Value Entropy"] = round(sv_entropy, 4)
    results["Singular Value Variance"] = round(sv_variance, 4)

    # Space structure
    results["Isotropy Score"] = round(isotropy_score(embeddings), 4)
    results["KL Divergence to Gaussian"] = round(kl_to_gaussian(embeddings), 4)
    results["Mutual Information"] = round(mutual_info(embeddings), 4)
    results["Covariance Matrix Entropy"] = round(covariance_entropy(embeddings), 4)
    results["Silhouette Score"] = round(compute_silhouette_score(embeddings), 4)

    return results


# ============================================================
# CSV Output
# ============================================================
CSV_HEADER = [
    "Dataset",
    "Model",
    "Embedding Variance",
    "Cosine Similarity",
    "Mutual Distance Mean",
    "Top 5 Singular Values",
    "Singular Value Entropy",
    "Singular Value Variance",
    "Isotropy Score",
    "KL Divergence to Gaussian",
    "Mutual Information",
    "Covariance Matrix Entropy",
    "Silhouette Score",
]


def save_results_to_csv(all_results: list, filename: str = OUTPUT_CSV) -> None:
    """Save all evaluation results to CSV."""
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in all_results:
            writer.writerow(row)
    print(f"\nResults saved to {filename}")


# ============================================================
# Main
# ============================================================
def main():
    all_results = []

    for dataset in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset}")
        print(f"{'=' * 60}")

        for model in MODELS:
            print(f"\n------ {model} ------")

            try:
                embeddings = load_item_embedding(model, dataset)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue
            except KeyError as e:
                print(f"  SKIP: {e}")
                continue

            print(f"  Embedding shape: {embeddings.shape}  device: {embeddings.device}")

            metrics = evaluate_embeddings(embeddings)

            print(f"  Embedding Variance:        {metrics['Embedding Variance']}")
            print(f"  Cosine Similarity:          {metrics['Cosine Similarity']}")
            print(f"  Mutual Distance Mean:       {metrics['Mutual Distance Mean']}")
            print(f"  Top 5 Singular Values:      {metrics['Top 5 Singular Values']}")
            print(f"  Singular Value Entropy:     {metrics['Singular Value Entropy']},  "
                  f"Variance: {metrics['Singular Value Variance']}")
            print(f"  Isotropy Score:             {metrics['Isotropy Score']}")
            print(f"  KL Divergence to Gaussian:  {metrics['KL Divergence to Gaussian']}")
            print(f"  Mutual Information:         {metrics['Mutual Information']}")
            print(f"  Covariance Matrix Entropy:  {metrics['Covariance Matrix Entropy']}")
            print(f"  Silhouette Score:           {metrics['Silhouette Score']}")

            row = [dataset, model] + [metrics[h] for h in CSV_HEADER[2:]]
            all_results.append(row)

            # Free GPU memory after each model evaluation
            if DEVICE.type == "cuda":
                del embeddings
                torch.cuda.empty_cache()

    save_results_to_csv(all_results)

    # Summary table
    print(f"\n{'=' * 110}")
    print("  Summary  (↑ higher better / ↓ lower better)")
    print(f"{'=' * 110}")
    print(f"{'Dataset':<10} {'Model':<12} {'Top5 SV':>30} {'SV Var':>8}↑ {'SV Ent':>8}↓ {'Cov Ent':>8}↓ {'Isotropy':>9}↓ {'KL2Gauss':>9}")
    print("-" * 110)
    for row in all_results:
        ds, md = row[0], row[1]
        sv5, sv_var, sv_ent, cov_ent, iso, kl = row[5], row[7], row[6], row[11], row[8], row[9]
        print(f"{ds:<10} {md:<12} {str(sv5):>30} {str(sv_var):>8} {str(sv_ent):>8} {str(cov_ent):>8} {str(iso):>9} {str(kl):>9}")
    print("-" * 110)


if __name__ == "__main__":
    main()
