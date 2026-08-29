# Batch Normalization in VGG Networks

An empirical study of how Batch Normalization changes optimisation, gradient
behaviour, and test performance in VGG-style networks on CIFAR-10.

## Experiments

- Standard VGG versus VGG + BatchNorm.
- Learning rates from 1e-4 to 2e-3.
- Activation functions, loss functions, and optimisers.
- Gradient variance, predictiveness, learned filters, and loss-landscape
  visualisations.

## Structure

```text
.
├── code/
│   ├── functions.py
│   ├── run_all_experiments.py
│   └── visualization.py
├── configs/
├── results/                  # compact JSON summaries and figures
└── docs/final_report.pdf
```

Model checkpoints are intentionally excluded. They added hundreds of
megabytes and can be regenerated from the configs and scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python code/run_all_experiments.py
```

The first run downloads CIFAR-10 through torchvision. GPU acceleration is
recommended for the full experiment grid.

## Report

See [`docs/final_report.pdf`](docs/final_report.pdf) for the methodology,
results, and interpretation.

