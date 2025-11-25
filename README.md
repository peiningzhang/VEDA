# VEDA

This repository contains the reference implementation of **VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing** (AAAI 2026 submission #8622). VEDA integrates a variance-exploding (VE) diffusion schedule with simulated-annealing-inspired noise shaping inside an SE(3)-equivariant backbone, closing the gap between flow-level sampling speed and diffusion-level chemical fidelity. For a complete description of the VE schedule, annealing scheme, and arcsin-based SNR allocation, see the preprint:

- [VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing](https://arxiv.org/abs/2511.09568)

Compared to the MiDi baseline, VEDA adds residual preconditioning to better interface coordinate-predicting networks with diffusion objectives, achieving state-of-the-art valency stability and a median relaxation energy drop of 1.72 kcal/mol on GEOM-DRUGS while keeping sampling to 100 steps, as reported in the paper.

## Repository status

**Update (July 2024):** Several historical checkpoints vanished when the original drive account was deleted. If you have local copies, please reach out at `vignac.clement@gmail.com` or open an issue so we can re-host them.

## Installation

This code was tested with PyTorch 2.0.1, CUDA 11.8, and torch_geometric 2.3.1 on multi-GPU machines.

- Download anaconda/miniconda if needed.
- Create the environment that already bundles RDKit:

  ```bash
  conda create -c conda-forge -n midi rdkit=2023.03.2 python=3.9
  conda activate midi
  ```

- Verify RDKit:

  ```bash
  python3 -c "from rdkit import Chem"
  ```

- Install CUDA toolkits matching your driver, e.g.

  ```bash
  conda install -c "nvidia/label/cuda-11.8.0" cuda
  ```

- Install PyTorch (adjust if your CUDA version differs):

  ```bash
  pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install the remaining dependencies and set up the package:

  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```

## Datasets

We reuse MiDi’s preprocessing pipeline. Follow their instructions for downloading, preprocessing, and structuring the datasets (see [MiDi README](https://github.com/cvignac/MiDi/blob/master/README.md)). Processed files should live under the same hierarchy (e.g. `MiDi/data/geom/raw/`) so that our scripts locate them without changes.

- QM9 should download automatically on first run.
- For GEOM-DRUGS, place the raw pickles in `MiDi/data/geom/raw/`:
  - Train: https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle
  - Validation: https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle
  - Test: https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle

## Running QM9 experiments

For the QM9 “smol” setup discussed in the paper, we launch training with:

```bash
python train.py \
  --data_path data/qm9/smol/ \
  --dataset qm9 \
  --epochs 300 \
  --use_edm_mask_step \
  --mask_rate_strategy edm \
  --optimal_transport None \
  --val_check_epochs 10 \
  --warm_up_steps 2000 \
  --bond_loss_weight 0.5 \
  --use_cat_time_based_weight
```

This command activates the EDM-style mask schedule, annealed VE noise, and categorical time-weighting described in [the VEDA paper](https://arxiv.org/abs/2511.09568), providing the configuration we use to report QM9 metrics. Adjust epochs, validation cadence, or transport settings as needed for GEOM-DRUGS or ablation studies.

## Running GEOM-DRUGS experiments

For GEOM-DRUGS we keep the same EDM-inspired masking but shorten the validation cadence:

```bash
python train.py \
  --data_path data/geom-drugs/smol/ \
  --dataset geom-drugs \
  --use_edm_mask_step \
  --mask_rate_strategy edm \
  --optimal_transport None \
  --use_cat_time_based_weight \
  --val_check_epochs 5
```

This configuration matches the GEOM-DRUGS runs described in the preprint and serves as the starting point for all ablations on that dataset.