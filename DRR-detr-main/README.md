# DRR-DETR

This repository contains the DRR-DETR model implementation, training configurations, and lightweight entry-point scripts.

## Contents

- `src/`: model, data-loading, optimization, and training code
- `configs/`: RT-DETR and dataset YAML configurations
- `tools/train.py`: training and evaluation entry point
- `tools/infer.py`: image inference entry point
- `tools/export_onnx.py`: ONNX export entry point

Datasets, checkpoints, generated outputs, logs, and experiment records are intentionally not included. Put local datasets under `data/` (or update the paths in `configs/dataset/`) and keep checkpoints outside the repository or under an ignored `checkpoints/` directory.

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands from the repository root so that relative configuration paths resolve correctly.

## Usage

Train or evaluate with a YAML configuration:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r101vd_6x_coco.yml
python tools/train.py -c configs/rtdetr/rtdetr_r101vd_6x_coco.yml -r /path/to/checkpoint.pth --test-only
```

Run inference with a trained checkpoint:

```bash
python tools/infer.py -c configs/rtdetr/rtdetr_r101vd_6x_coco.yml -r /path/to/checkpoint.pth -f /path/to/image.jpg
```

Export a trained model to ONNX:

```bash
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r101vd_6x_coco.yml -r /path/to/checkpoint.pth -f model.onnx --check
```

The default dataset YAML files use `./data/...` paths as placeholders. Adjust them to match the local COCO-format dataset before training.
