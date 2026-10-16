<div align="center">

<h1>BBDRec: Brownian Bridge Diffusion<br>for Sequential Recommendation</h1>

<p>
  <a href="https://orcid.org/0009-0008-8874-9409">Yimeng Bai</a><sup>1</sup> ·
  <a href="https://orcid.org/0000-0002-7863-5183">Yang Zhang</a><sup>2,*</sup> ·
  <a href="https://orcid.org/0000-0003-1796-8504">Sihao Ding</a><sup>3</sup> ·
  <a href="https://orcid.org/0009-0003-0150-0445">Shaohui Ruan</a><sup>3</sup><br>
  <a href="https://orcid.org/0009-0006-7322-1578">Han Yao</a><sup>3</sup> ·
  <a href="https://orcid.org/0009-0009-3540-8608">Danhui Guan</a><sup>3</sup> ·
  <a href="https://orcid.org/0000-0002-5828-9842">Fuli Feng</a><sup>1</sup> ·
  <a href="https://orcid.org/0000-0001-6097-7807">Tat-Seng Chua</a><sup>2</sup>
</p>

<p>
  <sup>1</sup> University of Science and Technology of China<br>
  <sup>2</sup> National University of Singapore &nbsp; <sup>3</sup> ByteDance<br>
  <sup>*</sup> Corresponding author: <a href="mailto:zyang1580@gmail.com">Yang Zhang</a>
</p>

<p>
  <a href="https://arxiv.org/abs/2507.06121"><img src="https://img.shields.io/badge/Paper-arXiv%3A2507.06121-b31b1b" alt="Paper"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C" alt="PyTorch 2.1.2"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"></a>
</p>

<p><strong>Official PyTorch implementation, datasets, experiment scripts, and figure sources.</strong></p>

</div>

[Overview](#overview) · [Results](#results) · [Setup](#setup) · [Data](#data) · [Training](#training) · [Experiments](#experiments) · [Analysis](#analysis) · [Citation](#citation)

## Overview

BBDRec adapts diffusion models to sequential recommendation at **two levels**, combining expressive generative modeling and iterative prediction refinement with item discrimination.

1. **Transition formulation.** An established Brownian bridge connects target-item and user-history representations with a simple linear formulation and closed-form transitions. The forward process transforms the target representation toward user history; reverse generation starts from that history representation. This gives an **item ↔ history** transition.
2. **Model learning.** A progressive learning strategy first establishes discriminative item embeddings through **recommendation pretraining**, then uses them to initialize **diffusion-based fine-tuning with joint reconstruction and recommendation objectives**. The embeddings support both target representation generation and matching generated outputs to actual items.

At inference, BBDRec generates a target-item representation from user history and retrieves the top-K items through inner-product matching against the item embedding table. It builds on established diffusion bridge principles and focuses on their adaptation to recommendation.

The repository includes six preprocessed datasets, Stage-1 pretrained item embeddings, 11 baselines, and scripts for training, tuning, ablations, and representation analysis. See the [arXiv preprint](https://arxiv.org/abs/2507.06121) for the paper.

## Results

The current manuscript evaluates recommendations under the **full-ranking protocol**. The table below reports BBDRec's HR@20 and NDCG@20 as averages across five runs, in percent. Results at K = 5 and 10 are provided in the manuscript appendix.

| Dataset | HR@20 (%) | NDCG@20 (%) |
|---|---:|---:|
| Baby | 7.4841 | 3.2566 |
| Beauty | 17.1476 | 8.2508 |
| ML-100K | 22.9676 | 8.4818 |
| Sports | 7.9144 | 3.3872 |
| Toys | 12.1264 | 5.8464 |
| Yelp | 7.5349 | 3.0242 |

BBDRec generally outperforms the compared sequential and diffusion-based baselines: it achieves the highest HR@20 on all six datasets and the highest NDCG@20 on five. NDCG@20 on Toys is the exception.

## Setup

Clone the repository and create the environment:

```bash
git clone https://github.com/baiyimeng/BBDRec.git
cd BBDRec
conda create -n bbdrec python=3.10 -y
conda activate bbdrec
pip install -r requirements.txt
```

Dependencies are pinned in [requirements.txt](requirements.txt), including PyTorch 2.1.2 and NumPy 1.26.2. Training defaults to `cuda:0`; use `--device cpu` for a CPU run.

## Data

Datasets are stored in `datasets/data/<dataset>/dataset.pkl`. The manuscript uses 5-core filtering and a chronological leave-one-out split: the last interaction is held out for testing, the second-last for validation, and the rest for training. Counts below include all three splits.

| Dataset key | Users | Items | Interactions |
|---|---:|---:|---:|
| `baby` | 11,761 | 4,731 | 92,823 |
| `beauty` | 10,553 | 6,086 | 94,148 |
| `ml-100k` | 938 | 1,008 | 54,413 |
| `sports` | 22,686 | 12,301 | 185,718 |
| `toys` | 11,268 | 7,309 | 95,420 |
| `yelp` | 136,346 | 64,669 | 1,856,992 |

Each pickle contains three lists of item sequences:

```python
{
    "train": [[2588, 1310, 1], ...],
    "val":   [[1082], ...],
    "test":  [[1283], ...],
}
```

The same list index identifies the same user. Validation and test each contain one held-out item per user. Item IDs start at 1; 0 is reserved for padding. Loaders retain the most recent `max_len` history items and pad on the left. Model-specific training layouts are selected in `src/core/trainer.py::choose_model`.

## Training

Run training and experiment commands from `src/`, where configuration and checkpoint paths are resolved:

```bash
cd src
```

### BBDRec

**Stage 1: Recommendation Pretraining.** Train a sequence encoder and item embeddings with a full-softmax cross-entropy objective. Stage-1 checkpoints are included at `saved/pretrain/<dataset>/pretrain.pth`; to regenerate one:

```bash
python main.py --model pretrain --dataset ml-100k --device cuda:0
```

**Stage 2: Diffusion-Based Fine-Tuning.** Initialize the item embedding table from the selected Stage-1 checkpoint. The Stage-2 sequence encoder and denoiser are initialized separately. Train with the joint objective `L = L_rec + η L_diff`, where `L_rec` is full-softmax cross-entropy and `L_diff` is target-representation MSE. For ML-100K:

```bash
python main.py --model bbdrec --dataset ml-100k --device cuda:0 \
    --diffusion_steps 4 --m 1 --loss_scale 10 --lambda_uncertainty 0.001 \
    --description bbdrec_ml-100k
```

The recommendation objectives in both stages use the full item set, without negative sampling or sampled softmax. The training schedule is implemented in [trainer.py](src/core/trainer.py). Checkpoints are saved to `saved/<model>/<dataset>/<description>.pth`; pretraining always uses the filename `pretrain.pth`. The bundled weights are Stage-1 checkpoints; final model checkpoints are produced by training.

The manuscript selects the following configurations on the validation set. These settings are also used by [reproduce_bbdrec.sh](src/reproduce_bbdrec.sh):

| Dataset | `--diffusion_steps` (T) | `--m` | `--loss_scale` (η) | `--lambda_uncertainty` (λ) |
|---|---:|---:|---:|---:|
| `baby` | 2 | 8 | 1 | 0.001 |
| `beauty` | 8 | 1 | 1 | 0.001 |
| `ml-100k` | 4 | 1 | 10 | 0.001 |
| `sports` | 2 | 1 | 0.01 | 0.001 |
| `toys` | 16 | 8 | 1 | 0.001 |
| `yelp` | 2 | 8 | 0.1 | 0.001 |

The bridge variance is `delta_t = m * beta_t * (1 - beta_t)`; `m` is the variance scale, `T` is the number of diffusion steps, `η` weights reconstruction, and `λ` controls the noisy-state and timestep inputs relative to user history.

### Baselines

Use the same entry point with a model key:

```bash
python main.py --model sasrec --dataset ml-100k --device cuda:0 \
    --description sasrec_ml-100k
```

| Family | Model keys |
|---|---|
| Recurrent / variational | `gru4rec`, `svae` |
| Transformer / attention | `sasrec`, `bert4rec`, `lightsans`, `core`, `fearec`, `eulerformer` |
| Diffusion | `diffurec`, `dreamrec`, `sdifrec` |

All models use the shared training interface. Adaptations are documented in `src/models/` and `choose_model`; for example, the FEARec port omits its dataset-level contrastive loss, while SVAE and EulerFormer expose auxiliary losses to the trainer.

### Configuration and outputs

[config.yaml](src/config.yaml) defines defaults. Supported command-line flags override those values; run `python main.py --help` to list them. Model-specific options without CLI flags can be set in YAML.

Common defaults are `hidden_size=128`, `max_len=50`, `batch_size=512`, `lr=0.001`, `dropout=0.1`, and `emb_dropout=0.3`. Both stages use Adam with cosine annealing, run for up to 500 epochs, validate every 5 epochs, and stop after 4 validation rounds without improvement. The default sequence encoder and denoiser each use two Transformer layers. Evaluation reports HR and NDCG at K = 5, 10, and 20.

Each training command launches one run. Use `--random_seed` and a distinct `--description` to record separate runs; a single command does not produce the five-run mean reported above. Single runs print results to the terminal. Batch scripts capture logs and summarize HR@20/NDCG@20 under `logs/`.

## Experiments

Before using a batch script, set `NUM_GPUS`, `TASKS_PER_GPU`, and `GPU_START` near its top to match the available devices. The scripts use Bash 4.3+ and GNU grep on Linux. BBDRec runs require a Stage-1 checkpoint for each selected dataset.

| Command (from `src/`) | Purpose |
|---|---|
| `bash reproduce_bbdrec.sh` | Train BBDRec on all six datasets with the configurations above. |
| `bash run_baselines.sh` | Run 11 baselines across six datasets. |
| `bash tune_bbdrec_ds_m_ls.sh` | Search the T × m × η grid for the dataset selected in the script. |
| `bash ablation_bbdrec.sh` | Run the variants configured in the helper script. |

The tuning helper searches 100 combinations: T ∈ {2, 4, 8, 16, 32}, m ∈ {1, 2, 4, 8, 16}, and η ∈ {0.01, 0.1, 1, 10}. It keeps λ at the YAML default of 0.001. The manuscript additionally examines λ ∈ {0.001, 0.01, 0.1, 1}.

The manuscript studies six variants: **w/o BB**, **w/o pretrain**, **w/ frozen embeddings**, **w/o MSE**, **w/o CE**, and **w/ MLP denoiser**. Together, they examine transition formulation, progressive learning, the reconstruction/recommendation objectives, and denoiser architecture. The paper also applies progressive learning to DreamRec and DiffuRec.

Inspect the `ABLATIONS` list in the batch helper before launching it; its configured runs are not a one-to-one list of the manuscript's six variants. Available switches include `bbdrec-0` for training without pretraining, `--loss_scale 0` for removing MSE, and `--diff_decoder mlp` for the MLP denoiser.

## Analysis

| Location | Purpose |
|---|---|
| [trajectory_viz.py](src/analysis/trajectory_viz.py) | Visualize forward Markov trajectories using the encoded history endpoint and independent noise innovations. |
| [trajectory_sampling.py](src/analysis/trajectory_sampling.py) | Forward transition samplers for Brownian bridges and DDPM. |
| [emb_viz.py](src/analysis/emb_viz.py) | Visualize item embeddings with t-SNE and density contours. |
| [emb_metrics.py](src/analysis/emb_metrics.py) | Compute statistics of the learned embedding distributions. |
| [src/figure/](src/figure/) | Figure PDFs, editable PowerPoint files, and plotting scripts. |

Model-based analyses require final trained checkpoints, not just the bundled Stage-1 weights. The plotting scripts in `src/figure/` render recorded arrays and do not launch training.

For trajectory visualization, provide the actual run settings for each compared model. From `src/`:

```bash
python analysis/trajectory_viz.py --dataset baby \
    --run-configs /path/to/actual_run_configs.yaml \
    --saved-dir /path/to/final_checkpoints
```

See the [trajectory visualization guide](src/analysis/trajectory_viz.md) for the configuration format and expected checkpoint names. The script saves the high-dimensional trajectories, t-SNE coordinates, and metadata alongside the SVG plots. Existing manuscript figures are not regenerated by running the sampling tests.

Run the eight NumPy-based sampling checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_trajectory_sampling.py' -v
```

## Citation

```bibtex
@article{bai2025bbdrec,
  title   = {Brownian Bridge Diffusion for Sequential Recommendation},
  author  = {Bai, Yimeng and Zhang, Yang and Ding, Sihao and Ruan, Shaohui and
             Yao, Han and Guan, Danhui and Feng, Fuli and Chua, Tat-Seng},
  journal = {arXiv preprint arXiv:2507.06121},
  year    = {2025}
}
```

## Acknowledgments

We thank the authors of [RecBole](https://github.com/RUCAIBox/RecBole), [DreamRec](https://github.com/YangZhengyi98/DreamRec), [SVAE](https://github.com/noveens/svae_cf), and [EulerFormer](https://github.com/Ethan-TZ/EulerFormer) for their open-source implementations used as references for the baseline ports.
