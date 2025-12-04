# VEDA

This repository contains the reference implementation of **VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing** (AAAI 2026 submission #8622). VEDA integrates a variance-exploding (VE) diffusion schedule with simulated-annealing-inspired noise shaping inside an SE(3)-equivariant backbone, closing the gap between flow-level sampling speed and diffusion-level chemical fidelity. For a complete description of the VE schedule, annealing scheme, and arcsin-based SNR allocation, see the preprint:

- [VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing](https://arxiv.org/abs/2511.09568)

Compared to the MiDi baseline, VEDA adds residual preconditioning to better interface coordinate-predicting networks with diffusion objectives, achieving state-of-the-art valency stability and a median relaxation energy drop of 1.72 kcal/mol on GEOM-DRUGS while keeping sampling to 100 steps, as reported in the paper.

## Models and Datasets

You can either download pre-trained models and processed datasets directly from:
- [Google Drive: Models and Data](https://drive.google.com/drive/folders/1-1jYJ1EKOUX6FA5NrcKUAcbSdB-1mhXu?usp=sharing)

Or process the datasets yourself:

### QM9

We copied the code from [MiDi](https://github.com/cvignac/MiDi) to download the QM9 dataset and create the data splits. We provide the code to do this, as well as create the Smol internal dataset representation used for training in the `notebooks/qm9.ipynb` notebook.

### GEOM-DRUGS

For GEOM-DRUGS we also follow the URLs provided in the MiDi repo. GEOM-DRUGS is preprocessed using the `preprocess.py` script. GEOM-DRUGS URLs from MiDi are as follows:

- Train: https://drive.switch.ch/index.php/s/UauSNgSMUPQdZ9v
- Validation: https://drive.switch.ch/index.php/s/YNW5UriYEeVCDnL
- Test: https://drive.switch.ch/index.php/s/GQW9ok7mPInPcIo

After downloading the raw pickle files, place them in a `raw` folder and run:
```bash
python preprocess.py --data_path path/to/geom-drugs --raw_data_folder raw --save_data_folder smol
```

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

## Running QM9 experiments

For the QM9 "smol" setup discussed in the paper, we launch training with:

```bash
python train.py \
  --data_path data/qm9/smol/ \
  --dataset qm9 \
  --epochs 300 \
  --mask_rate_strategy edm \
  --optimal_transport None \
  --warm_up_steps 2000 \
  --bond_loss_weight 0.5 \
  --use_cat_time_based_weight \
  --x_pred_mode constant
```

This command activates the EDM-style mask schedule, annealed VE noise, and categorical time-weighting described in [the VEDA paper](https://arxiv.org/abs/2511.09568), providing the configuration we use to report QM9 metrics. Adjust epochs, validation cadence, or transport settings as needed for GEOM-DRUGS or ablation studies.

### Evaluation

To evaluate a trained QM9 model:

```bash
python evaluate.py \
  --data_path path/data/qm9/smol/ \
  --ckpt_path qm9/checkpoints/last.ckpt \
  --dataset qm9 \
  --coord_noise_std_dev 0.4 \
  --temperature 0.9 \
  --adaptive_cat_noise_level
```

## Running GEOM-DRUGS experiments

For GEOM-DRUGS we keep the same EDM-inspired masking but shorten the validation cadence:

```bash
python train.py \
  --data_path data/geom-drugs/smol/ \
  --dataset geom-drugs \
  --mask_rate_strategy edm \
  --optimal_transport None \
  --val_check_epochs 10 \
  --x_pred_mode adaptive
```

This configuration matches the GEOM-DRUGS runs described in the preprint and serves as the starting point for all ablations on that dataset.

### Evaluation

To evaluate a trained GEOM-DRUGS model:

```bash
python evaluate.py \
  --data_path data/geom-drugs/smol/ \
  --ckpt_path model/last.ckpt \
  --dataset geom-drugs \
  --coord_noise_std_dev 0.4 \
  --sampling_scheduler_factor_rho 2.5 \
  --mask_rate_strategy edm \
  --adaptive_cat_noise_level
```

### GEOM-DRUG-revisited Benchmark

For GEOM-DRUG-revisited Benchmark evaluation, see our parallelized implementation: [geom-drugs-3dgen-evaluation](https://github.com/peiningzhang/geom-drugs-3dgen-evaluation).

To run the benchmark:
1. First, generate molecules using `predict.py` with the same parameters as `evaluate.py`:
   ```bash
   python predict.py \
     --data_path data/geom-drugs/smol/ \
     --ckpt_path model/last.ckpt \
     --dataset geom-drugs \
     --coord_noise_std_dev 0.4 \
     --sampling_scheduler_factor_rho 2.5 \
     --mask_rate_strategy edm \
     --adaptive_cat_noise_level
   ```
2. Then run the energy benchmark:
   ```bash
   bash -x run_energy_benchmark.sh path/of/sdf
   ```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2025veda,
  title={VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing},
  author={Zhang, Peining and Bi, Jinbo and Song, Minghu},
  journal={arXiv preprint arXiv:2511.09568},
  year={2025}
}
```