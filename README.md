# RGQ-DETR: Geometry-Conditioned Detection of Live and DeadSeedlings after Black Cutworm Exposure Geometry-Conditioned Detection

## Introduction
Based on the RT-DETR framework, this study proposes Representation–Geometry–Quality DETR (RGQ-DETR), an end-to-end detector designed for post-exposure live/dead seedling assessment. The method follows the agricultural decision chain: retain the fine structural evidence that indicates stem support and ground contact, guide feature sampling toward the evolving plant geometry, and train classification quality together with localization reliability. 

![image](https://github.com/xwhh123/detr/blob/main/tools/faf67d767344f54363c7b42e4a597098.png)

## Get Started

### 1. Prerequisites

**Recommended environment**

- Ubuntu >= 20.04
- CUDA >= 11.8
- Python == 3.9.7
- CUDA-compatible PyTorch/torchvision pair (tested with PyTorch 2.0.1 and torchvision 0.15.2)

The repository provides the complete dependency list in `requirements.txt`. It intentionally leaves the PyTorch versions unpinned; record the resolved versions for exact reproduction. The dataset and checkpoints are not included.

**Step 0. Create the Conda environment**

~~~bash
conda create --name rgq-detr python=3.9.7 -y
conda activate rgq-detr
~~~

**Step 1. Install dependencies from `requirements.txt`**

~~~bash
pip install -r requirements.txt
~~~

### 2. Prepare the Dataset

The model expects a COCO-format dataset with two categories:
~~~bash
0: live_plant
1: dead_plant
~~~
Place the dataset under data/so/:
~~~bash
data/so/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
~~~
### 3. Training

**Single GPU**

~~~bash
python tools/train.py \
  -c configs/rtdetr/rgq.yml \
  --amp
~~~

**Multi GPU**

The R101 configuration uses batch size 3 per GPU. The following four-GPU command gives a global batch size of 12:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 --master-port=8989 \
  tools/train.py \
  -c configs/rtdetr/rgq.yml \
  --amp
~~~

Training outputs are written to the configured output_dir, including:
~~~bash
best.pth
last.pth
checkpointXXXX.pth
log.txt
eval/
~~~

