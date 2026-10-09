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

[Overview](#overview) · [Setup](#setup) · [Data](#data) · [Training](#training) · [Experiments](#experiments) · [Analysis](#analysis) · [Citation](#citation)

## Overview

BBDRec models sequential recommendation as a **preference-bridging process between user history and the target item**, using a linear Brownian bridge in the learned representation space.

- **Brownian bridge:** connect target-item and user-history representations through a stochastic diffusion process.
- **Progressive learning:** pretrain discriminative item embeddings, then jointly optimize recommendation and diffusion objectives.
- **Recommendation:** generate a target-item representation from user history through reverse diffusion, then retrieve items by inner-product matching.

The repository includes six preprocessed datasets, pretrained item embeddings, 11 baselines, and scripts for training, tuning, ablation studies, and representation analysis. The [paper](https://arxiv.org/abs/2507.06121) provides the theoretical derivations and experimental results.

## Setup

From the repository root:

```bash
conda create -n bbdrec python=3.10 -y
conda activate bbdrec
pip install -r requirements.txt
```

Dependencies are pinned in [requirements.txt](requirements.txt), including PyTorch 2.1.2 and NumPy 1.26.2. Training defaults to `cuda:0`; use `--device cpu` for a CPU run.

## Data

Datasets are stored in `datasets/data/<dataset>/dataset.pkl`. Counts below include train, validation, and test interactions.

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

Stage-1 checkpoints are included at `saved/pretrain/<dataset>/pretrain.pth`. To regenerate one:

```bash
python main.py --model pretrain --dataset ml-100k --device cuda:0
```

Train Stage 2 with the ML-100K settings:

```bash
python main.py --model bbdrec --dataset ml-100k --device cuda:0 \
    --diffusion_steps 4 --m 1 --loss_scale 10 --lambda_uncertainty 0.001 \
    --description bbdrec_ml-100k
```

Stage 2 loads the pretrained item embeddings, freezes them for the first five epochs, and then jointly updates all components. Checkpoints are saved to `saved/<model>/<dataset>/<description>.pth`; pretraining always uses the filename `pretrain.pth`.

Final configurations in `reproduce_bbdrec.sh`:

| Dataset | `--diffusion_steps` (T) | `--m` | `--loss_scale` (η) | `--lambda_uncertainty` (λ) |
|---|---:|---:|---:|---:|
| `baby` | 2 | 8 | 1 | 0.001 |
| `beauty` | 8 | 1 | 1 | 0.001 |
| `ml-100k` | 4 | 1 | 10 | 0.001 |
| `sports` | 2 | 1 | 0.01 | 0.001 |
| `toys` | 16 | 8 | 1 | 0.001 |
| `yelp` | 2 | 8 | 0.1 | 0.001 |

The variance coefficient follows the paper directly: `delta_t = m * beta_t * (1 - beta_t)`.

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

Common defaults are `hidden_size=128`, `max_len=50`, `batch_size=512`, `lr=0.001`, `dropout=0.1`, and `emb_dropout=0.3`. Training runs for up to 500 epochs, with validation every 5 epochs and early stopping after 4 validation rounds without improvement. Evaluation reports HR and NDCG at K = 5, 10, and 20.

Single runs print results to the terminal. Batch scripts capture logs and summarize HR@20/NDCG@20 under `logs/`. Set `--description` to distinguish checkpoints across runs.

## Experiments

Before using a batch script, set `NUM_GPUS`, `TASKS_PER_GPU`, and `GPU_START` near its top to match the available devices. The scripts use Bash 4.3+ and GNU grep on Linux. BBDRec runs require a Stage-1 checkpoint for each selected dataset.

| Command (from `src/`) | Purpose |
|---|---|
| `bash reproduce_bbdrec.sh` | Train BBDRec on all six datasets with the configurations above. |
| `bash run_baselines.sh` | Run 11 baselines across six datasets. |
| `bash tune_bbdrec_ds_m_ls.sh` | Search the T × m × η grid for the dataset selected in the script. |
| `bash ablation_bbdrec.sh` | Run the configured ablations and compare them with BBDRec. |

The tuning grid contains 100 combinations: T ∈ {2, 4, 8, 16, 32}, m ∈ {1, 2, 4, 8, 16}, and η ∈ {0.01, 0.1, 1, 10}. The script keeps λ at the YAML default of 0.001.

The ablation script includes `no_pretrain` (`bbdrec-0`), `no_warmup` (`bbdrec-1`), `no_mse` (`--loss_scale 0`), and `mlp_decoder` (`--diff_decoder mlp`).

## Analysis

| Location | Purpose |
|---|---|
| `src/analysis/trajectory_viz.py` | Visualize forward diffusion trajectories. |
| `src/analysis/emb_viz.py` | Visualize item embeddings with t-SNE or PCA. |
| `src/analysis/emb_metrics.py` | Compute embedding-distribution statistics. |
| `src/figure/` | Figure PDFs, editable PowerPoint files, and Python plotting scripts. |

Analysis scripts require trained checkpoints under `src/saved/`; check their dataset/model selections and expected checkpoint filenames before running. The plotting scripts render recorded result arrays and do not launch experiments.

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
