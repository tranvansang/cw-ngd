# Overview

This repository contains the source code for the experiments of the paper "Enhanced Component-Wise Natural Gradient Descent Training Method for Deep Neural Networks".

Authors: Sang Van Tran, Toshiyuki Nakata, Rie Shigetomi Yamaguchi, Mhd Irvan, and Yoshihide Yoshimoto.

The source code of the experiments is available at https://github.com/tranvansang/cw-ngd

The experiment results are available at https://github.com/tranvansang/cw-ngd-results


# Setup
- Python version: `3.11.5`.
- Install pip packages:
```bash
pip install torch==2.1.1 torchvision==0.16.1 opt_einsum==3.3.0
```
- Download data:
```bash
python lib/data.py
```

# Run

## Experiment 1
```bash
python main_scan_params.py | tee logs/1-cross-hyperparameters.csv
```

If there is Out-Of-Memory error, change `_max_processes_per_gpu = 2`  to `_max_processes_per_gpu = 1` in `main_scan_params.py`.

## Experiment 2
```bash
scripts/2-strategies.sh
```

## Experiment 3
```bash
scripts/3-all-algorithms.sh
```
