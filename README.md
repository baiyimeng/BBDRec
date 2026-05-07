# BBDRec
[![arXiv](https://img.shields.io/badge/arXiv-2507.06121-red.svg)](https://arxiv.org/abs/2507.06121)

This is the PyTorch implementation of the paper:

> [**Unconditional Diffusion for Generative Sequential Recommendation**](https://arxiv.org/abs/2507.06121)
>
> Yimeng Bai, Yang Zhang, Sihao Ding, Shaohui Ruan, Han Yao, Danhui Guan, Fuli Feng, Tat-Seng Chua.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train BBDRec on a dataset (e.g., ml-100k)
python main.py --model bbdrec --dataset ml-100k

# Train baselines
bash run_baselines.sh
```

## Datasets

| Dataset  | #Items | #Interactions |
|----------|--------|---------------|
| ML-100K  | 1,008  | 100K          |
| Yelp     | 64,669 | 1.6M          |
| Sports   | 12,301 | 1.5M          |
| Baby     | 4,731  | 0.5M          |
| Toys     | 7,309  | 0.4M          |
| Beauty   | 6,086  | 0.2M          |