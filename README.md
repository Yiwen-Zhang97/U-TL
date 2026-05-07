# U-TL: Transfer Learning Framework for High-Resolution Urban Air Temperature Estimation

This repository contains the PyTorch implementation of the Urban Transfer Learning (U-TL) framework used to generate the U-HAT (Urban High-Resolution Air Temperature) dataset described in:

> Zhang, Y., Zhao, L., Chakraborty, T. C., Mazumdar, P., Zhang, K., & Gentine, P.  
> *Transfer learning reveals large discrepancies between air and land surface temperatures in cities.*

The framework leverages transfer learning to estimate high-resolution urban near-surface air temperature (Ta) from satellite observations, meteorological forcings, and urban surface characteristics.

---

# Repository Structure

```text
repo/
├── tl_helpers.py
├── train_pretrain.py
├── train_finetune.py
├── README.md
└── data/
```

---

# Overview

The U-TL framework consists of two stages:

## 1. Pretraining Stage

The model is first pretrained to predict satellite-observed land surface temperature (LST) using:

- Landsat-derived urban surface imagery
- Meteorological forcing variables
- Temporal information (month)

Because LST observations are spatially continuous and widely available, this stage enables the network to learn general land–atmosphere interaction patterns.

Implemented in:

```text
train_pretrain.py
```

---

## 2. Fine-Tuning Stage

The pretrained model is subsequently fine-tuned using sparse in-situ urban air temperature (Ta) observations.

Additional architectural modifications include:

- LST residual skip connection
- Reduced fully connected layers
- Fine-tuning-specific dropout configuration

Implemented in:

```text
train_finetune.py
```

---

# Main Features

- Transfer learning framework for urban climate applications
- Shared helper utilities across pretraining and fine-tuning
- Multi-modal inputs:
  - Satellite imagery
  - Reanalysis meteorological forcings
  - Urban morphology information
  - Temporal encoding
- PyTorch DataParallel support
- Configurable training and fine-tuning pipelines
- Modularized architecture and preprocessing utilities

---

# Input Data Structure

## Pretraining Data

Pretraining uses HDF5 files containing shuffled samples.

Each sample includes:

| Variable | Description |
|---|---|
| Month | Month index |
| Forcing variables | ERA5-derived meteorological forcings |
| LST | MODIS land surface temperature |
| Image channels | Landsat-derived imagery and urban descriptors |

### Image Channels

```python
[
    "Red",
    "Green",
    "Blue",
    "NIR",
    "SWIR1",
    "NDBI",
    "NDVI",
    "Elevation"
]
```

### Forcing Variables

```python
[
    "FLDS",
    "FSDS",
    "PRECTmms",
    "PSRF",
    "QBOT",
    "TBOT",
    "WIND_U",
    "WIND_V",
    "Building_Height"
]
```

---

## Fine-Tuning Data

Fine-tuning uses CSV files containing:

- Urban air temperature observations
- Satellite predictors
- Meteorological forcings
- Spatial metadata

---

# Model Architecture

The model uses a simplified residual convolutional neural network consisting of:

- Shared CNN backbone
- Meteorological forcing encoder
- Temporal one-hot embedding
- Fully connected regression head

The fine-tuning model additionally incorporates:

- LST residual skip connection
- Smaller fully connected layers
- Fine-tuning dropout adjustments

---

# Installation

## Requirements

- Python >= 3.9
- PyTorch
- torchvision
- numpy
- pandas
- h5py
- tqdm

Install dependencies:

```bash
pip install torch torchvision numpy pandas h5py tqdm
```

---

# Training

## Pretraining

Edit paths in:

```python
train_pretrain.py
```

Then run:

```bash
python train_pretrain.py
```

---

## Fine-Tuning

Edit paths in:

```python
train_finetune.py
```

Then run:

```bash
python train_finetune.py
```

---

# Path Configuration

The repository intentionally uses placeholder paths:

```python
PATH_TO_DATA = "/path/to/data"
DIR_TO_STORE = "/path/to/output"
```

Users should replace these with their local or HPC paths.

---

# Outputs

The training scripts save:

## Model Checkpoints

```text
best_resnet*.pt
```

## Training Records

```text
train_loss*.pkl
test_loss*.pkl
train_loss_l1*.pkl
test_loss_l1*.pkl
```

---

# Helper Utilities

Shared functionality is centralized in:

```text
tl_helpers.py
```

This includes:

- Dataset loaders
- Data preprocessing
- Normalization constants
- ResNet architectures
- Transfer learning utilities
- SuperLoss implementation
- Random seed initialization
- Pickle saving utilities

---

# Hardware Notes

The code supports:

- CPU training
- Single-GPU training
- Multi-GPU training using `torch.nn.DataParallel`

Example:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.DataParallel(model)
```

---

# Citation

If you use this repository or the associated dataset, please cite:

```text
Zhang, Y., Zhao, L., Chakraborty, T. C., Mazumdar, P., Zhang, K., & Gentine, P.
Transfer learning reveals large discrepancies between air and land surface temperatures in cities.
```

---

# License

This repository is provided for research and academic use.

Please contact the authors regarding commercial usage or redistribution.

---

# Contact

For questions regarding the repository, methodology, or dataset, please contact the corresponding authors of the associated publication.

