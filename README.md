# [cite_start]GNR638 Assignment 2: Pre-trained CNN Representation Transfer and Robustness Analysis [cite: 1, 2]


## Overview
[cite_start]This repository contains the reproducible codebase for evaluating the behavior of pre-trained convolutional neural network (CNN) backbones on a supervised image classification task[cite: 6]. [cite_start]The models are evaluated on the Aerial Images Dataset (AID) [cite: 33] [cite_start]across five controlled experimental scenarios[cite: 37]:
1. [cite_start]Linear Probe Transfer [cite: 41]
2. [cite_start]Fine-Tuning Strategies [cite: 53]
3. [cite_start]Few-Shot Learning Analysis [cite: 68]
4. [cite_start]Corruption Robustness Evaluation [cite: 81]
5. [cite_start]Layer-Wise Feature Probing [cite: 97]

**Selected Architectures:**
* [cite_start]ResNet50 [cite: 25]
* [cite_start]DenseNet121 [cite: 28]
* [cite_start]EfficientNet-B0 [cite: 29]

---

## [cite_start]1. Environment Setup [cite: 126, 127, 128]
This framework is built using Python 3.12 and PyTorch. 

**Install required dependencies:**
```bash
pip install torch torchvision torchaudio scikit-learn timm thop tqdm matplotlib Pillow numpy tensorboard
```
##Step 1: Create Data Splits

This step generates the 100%, 20%, and 5% few-shot training subsets and the validation split using the assigned group seed.
```bash
python create_dataset.py --seed 42
```

##Step 2: Train Models (Scenarios 4.1, 4.2, 4.3)

Run the following commands for each model.

Example models:

resnet50

densenet121

efficientnet_b0

Note

Full-data training: 30 epochs

Few-shot training: 20 epochs

Scenario 4.1 & 4.2

Linear Probe and Fine-Tuning Strategies (100% Data)
```bash
python train.py --model resnet50 --strategy linear_probe --train_split train_100 --epochs 30 --batch_size 32

python train.py --model resnet50 --strategy full --train_split train_100 --epochs 30 --batch_size 32

python train.py --model resnet50 --strategy last_block --train_split train_100 --epochs 30 --batch_size 32

python train.py --model resnet50 --strategy selective_20 --train_split train_100 --epochs 30 --batch_size 32
```

Scenario 4.3

Few-Shot Learning (20% and 5% Data)
```bash
python train.py --model resnet50 --strategy full --train_split train_20 --epochs 20 --batch_size 32

python train.py --model resnet50 --strategy full --train_split train_05 --epochs 20 --batch_size 32
```
##Step 3: Evaluate Linear Probe

(Embeddings & Confusion Matrix)

Generates:

t-SNE visualizations

Confusion matrices

Required for Scenario 4.1.
```bash
python eval_model.py \
--run_name resnet50_linear_probe_train_100 \
--model resnet50 \
--plot_embeddings \
--embed_method tsne
```

##Step 4: Corruption Robustness Evaluation (Scenario 4.4)

Evaluates fully fine-tuned models under image corruptions:

Gaussian Noise

Motion Blur

Brightness Shift
```bash
python robustness_test.py \
--run_names resnet50_full_train_100 densenet121_full_train_100 efficientnet_b0_full_train_100 \
--batch_size 32
```
##Step 5: Layer-Wise Feature Probing (Scenario 4.5)

This step:

Extracts intermediate feature representations

Trains linear classifiers on early, middle, and final layers

Generates PCA clustering plots
```bash
python probe_features.py --model resnet50 --batch_size 64
```
Outputs and Deliverables

All generated assets required for the technical report are automatically saved in the:

checkpoints/

directory.

Training Logs

TensorBoard logs

Loss / accuracy curves (.png)

Efficiency metrics

MACs / FLOPs

Parameter counts

Visualizations

Confusion matrices

t-SNE embeddings

PCA layer-wise feature scatter plots

Robustness Evaluation

CSV files tracking accuracy degradation

Text summaries containing:

Relative Robustness

Corruption Error formulas
