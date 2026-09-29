# RGQ-DETR: Geometry-Conditioned Detection of Live and DeadSeedlings after Black Cutworm Exposure Geometry-Conditioned Detection

## Introduction
Based on the RT-DETR framework, this study proposes Representation–Geometry–Quality DETR (RGQ-DETR), an end-to-end detector designed for post-exposure live/dead seedling assessment. The method follows the agricultural decision chain: retain the fine structural evidence that indicates stem support and ground contact, guide feature sampling toward the evolving plant geometry, and train classification quality together with localization reliability. 

![image](https://github.com/xwhh123/detr/blob/main/tools/faf67d767344f54363c7b42e4a597098.png)

## Get Started

### 2. Prepare the Dataset

The model expects a COCO-format dataset with two categories:
0: live_plant
1: dead_plant

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

~~~text
configs/rtdetr/ablation/
~~~
