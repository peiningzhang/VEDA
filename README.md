# VEDA
This repository contains code for the AAAI 2026 submission titled 'VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing' (Submission #8622).
# MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation

**Update (July 2024): My drive account has unfortunately been deleted, and I have lost access some checkpoints. If you happen to have a downloaded checkpoint stored locally, I would be glad if you could send me an email at vignac.clement@gmail.com or raise a Github issue.**

[Link to the paper](https://arxiv.org/abs/2302.09048)

Clément Vignac*, Nagham Osman*, Laura Toni, Pascal Frossard

ECML 2023
## Installation

This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometric 2.3.1 on multiple gpus.

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n midi rdkit=2023.03.2 python=3.9```
  - `conda activate midi`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```



## Datasets
We follow [MiDi](https://github.com/cvignac/MiDi/blob/master/README.md) for the data processing part. Please refer to their repository and instructions for details on obtaining, preprocessing, and formatting the datasets. The processed data should be organized in the same structure as MiDi expects, and placed in the appropriate folders (e.g., `MiDi/data/geom/raw/`). This ensures compatibility with our scripts for molecular generation.

  - QM9 should download automatically
  - For GEOM, download the data and put in `MiDi/data/geom/raw/`:
    - train: https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle
    - validation: https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle
    - test: https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle