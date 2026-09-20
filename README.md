# Real-Time NMR: AI-Enhanced 1D ¹H NMR with HSQC-Guided Lineshape Correction

Deep-learning (RH-Unet) based lineshape transfer from 2D ¹H–¹³C HSQC to 1D NOESY-presat spectra, enabling minute-scale metabolic monitoring of living cells by ¹H NMR.

## Overview

Three-dimensional (3D) cell cultures introduce magnetic field inhomogeneity that broadens ¹H NMR resonances, forcing a trade-off between rapid but poorly resolved 1D spectra and high-resolution but slow (~38 min) 2D HSQC experiments. This repository implements a two-step workflow:

1. **RH-Unet lineshape transfer**: A 1D U-Net convolutional neural network is trained at the first experimental time point to map Gaussian-like 2D HSQC metabolite signatures to the broadened Lorentzian peak shapes of 1D NOESY-presat (*noesygppr1d*) spectra.
2. **BVLS deconvolution**: Non-negative least-squares fitting against the RH-Unet-corrected reference signatures quantifies metabolite intensities at 2-min intervals without routine 2D acquisition.

The method was validated in HepG2 and MCF-7 cells (four biological replicates each) and applied to palmitic acid-induced lipotoxicity studies.

## Repository Structure

```
├── models_RHUnet.py          # RH-Unet model definitions (uNet8, uNet4, loss functions)
├── nmr_analysis.py           # NMR data loading, peak detection, and visualization utilities
├── cell_data_V_800M_MCF.ipynb  # Main processing: Bruker → RH-Unet → BVLS
├── validation_analysis.ipynb  # Validation: 1D vs 2D comparison + PA vs BSA statistics
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── data/                      # Processed CSV data (committed)
    ├── validation_hepg2/     # 4 HepG2 samples, 15 paired time points
    ├── validation_mcf7/       # 4 MCF-7 samples, 15 paired time points
    └── pa_experiment/         # 8 HepG2 samples (4 BSA + 4 PA), 450 time points
```

> **Note**: Raw Bruker data (~140 MB) is not included in the repository due to size. The `validation_analysis.ipynb` notebook runs entirely on the processed CSV files in `data/`.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12 (CUDA recommended for GPU training; CPU works for inference)
- nmrglue ≥ 0.9
- scipy
- numpy
- pandas
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The main notebook reads Bruker-format NMR experiments organized as:

```
data/
└── <experiment_name>/
    ├── 3/                    # HSQC (odd expnos: 3, 5, 7, ...)
    │   └── pdata/1/
    ├── 4/                    # NOESY-presat (even expnos: 4, 6, 8, ...)
    │   └── pdata/1/
    └── ...
```

Key acquisition parameters (800 MHz, 310 K):

| Parameter | 1D *noesygppr1d* | 2D *hsqcetgpsp.2* |
|---|---|---|
| Complex points | 32 K (FT 64 K) | 2048 × 64 (FT 4096 × 128) |
| Scans | 32 | 32 per increment |
| Relaxation delay | 2.0 s | 1.0 s |
| Acquisition time | ~2 min/spectrum | ~38 min/spectrum |

## Quick Start

### 1. Validation analysis (no Bruker data needed)

The `validation_analysis.ipynb` notebook runs on pre-processed CSV data and demonstrates:
- 1D vs 2D HSQC quantification comparison (Pearson r, Bland-Altman)
- Temporal profiles of 8 metabolite signals under BSA vs PA treatment
- Slope statistics between treatment groups

Simply open and run `validation_analysis.ipynb` — all required data is in `data/validation_hepg2/`, `data/validation_mcf7/`, and `data/pa_experiment/`.

### 2. Full processing workflow

To run the complete RH-Unet pipeline on raw Bruker data:

1. Place Bruker experiment directories under `data/`.
2. Open `cell_data_V_800M_MCF.ipynb` and run cells top-to-bottom.
3. Paths are configured in the first cell:

```python
from pathlib import Path
NB_DIR = Path.cwd()
DATA_DIR = NB_DIR / "data" / "<your_experiment_name>"
FACTOR_DIR = NB_DIR / "factors"
```

The main notebook covers:
- Bruker data loading via nmrglue
- Chemical shift axis definition
- 1D NOESY processing and visualization
- 2D HSQC slice extraction
- RH-Unet training and inference
- BVLS deconvolution of time-course spectra

## Citation

If you use this code, please cite:

> X. Xiao, S. He, B. Liu, X. Chai, C. Liu, M. Liu, B. Jiang*, X. Zhang*. Deep-Learning-Enhanced 1D ¹H NMR with HSQC-Guided Lineshape Correction for Minute-Scale Metabolic Monitoring in Living Cells.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- Xiongjie Xiao: xiaoxj007@outlook.com
