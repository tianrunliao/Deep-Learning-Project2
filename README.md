# VGG with Batch Normalization on CIFAR-10

**Undergraduate coursework.**

A PyTorch study of how Batch Normalization changes the training of a VGG-A
style network on CIFAR-10.

## Archived results

The only archived results are
[`results/multi_lr_experiment/standard_vgg/results.json`](results/multi_lr_experiment/standard_vgg/results.json)
and
[`results/multi_lr_experiment/vgg_batchnorm/results.json`](results/multi_lr_experiment/vgg_batchnorm/results.json).
They hold one 5-epoch run of a standard VGG and of a VGG with Batch
Normalization at each of four learning rates. Each run records per-epoch
training loss (`losses`), training accuracy (`train_accs`) and validation
accuracy (`val_accs`), plus gradient predictiveness (`gradient_pred`) and
gradient variance (`gradient_var`) at epochs 1, 3 and 5. The training script
that wrote these files is not in the repository, so the exact definitions of
the two gradient metrics are not recorded; `code/visualization.py` labels
predictiveness as a cosine similarity.

All numbers below are read directly from those two files.

Validation accuracy (%) after epoch 5:

| Learning rate | Standard VGG | VGG + BN |
|---|---:|---:|
| 0.0001 | 47.87 | 68.18 |
| 0.0005 | 56.79 | 74.70 |
| 0.001 | 9.92 | 74.26 |
| 0.002 | 9.43 | 69.92 |

Gradient metrics at epoch 5:

| Learning rate | Predictiveness, standard | Predictiveness, BN | Variance, standard | Variance, BN |
|---|---:|---:|---:|---:|
| 0.0001 | 0.671 | 0.081 | 4.3e-06 | 6.9e-06 |
| 0.0005 | 0.402 | 0.176 | 1.5e-06 | 2.0e-06 |
| 0.001 | 0.050 | 0.358 | 8.8e-10 | 1.6e-06 |
| 0.002 | 0.055 | 0.303 | 8.2e-10 | 5.2e-07 |

What the logs show:

1. VGG + BN has the higher validation accuracy after 5 epochs at every
   learning rate tested.
2. At learning rates 0.001 and 0.002 the standard VGG does not train: its
   training loss stays at 2.303 (ln 10) and its validation accuracy stays
   near chance (9.4% to 10.5%) for all 5 epochs. Its gradient variance there
   is about 1e-9, against roughly 1e-6 for VGG + BN.
3. Gradient predictiveness is higher for the standard VGG at the two small
   learning rates and higher for VGG + BN at the two large ones.

These are single runs of 5 epochs each, with no repeated seeds and no test
set evaluation, so they show early-training behaviour only.

## Code

| Path | Contents |
|---|---|
| `code/functions.py` | VGG-A with and without Batch Normalization, convolutional, residual and bottleneck blocks, Swish and Mish activations, loss and optimizer factories, CIFAR data loading and a trainer |
| `code/run_all_experiments.py` | Driver for activation-function, loss-function and optimizer comparisons, plus a text report generator |
| `code/visualization.py` | Plotting functions |
| `configs/` | Two YAML training configurations |

Limitations of the committed code:

1. The activation-function, loss-function and optimizer comparisons are
   implemented in `code/run_all_experiments.py`, but their results are not
   archived. The script also imports `train_model` from an
   `optimization_experiments` module that is not in the repository, so these
   comparisons cannot run as committed.
2. The report that `run_all_experiments.py` writes to `experiment_report.txt`
   is a hard-coded text template. Its numbers are not backed by any archived
   result.
3. `code/visualization.py` expects result files and model checkpoints that are
   not archived (for example `results/standard_vgg_results.json` and
   `results/checkpoints/`), so it cannot regenerate figures from the archived
   logs. No figures are included for that reason.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`prepare_data()` in `code/functions.py` downloads CIFAR-10 through
torchvision into `./data`. Model checkpoints (`*.pt`, `*.pth`) are not
committed.
