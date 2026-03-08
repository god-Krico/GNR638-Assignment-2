@echo off
set SEED=42

echo ========================================
echo Starting GNR638 Assignment 2 Pipeline
echo ========================================

echo Preparing dataset splits with seed %SEED%...
python create_dataset.py --seed %SEED%

for %%M in (resnet50 densenet121 efficientnet_b0) do (
    echo ========================================
    echo Processing Architecture: %%M
    echo ========================================

    echo [%%M] Training: Linear Probe...
    python train.py --model %%M --strategy linear_probe --train_split train_100 --epochs 30 --batch_size 32
    
    echo [%%M] Training: Full Fine-Tuning...
    python train.py --model %%M --strategy full --train_split train_100 --epochs 30 --batch_size 32
    
    echo [%%M] Training: Last Block Fine-Tuning...
    python train.py --model %%M --strategy last_block --train_split train_100 --epochs 30 --batch_size 32
    
    echo [%%M] Training: Selective 20%% Unfreezing...
    python train.py --model %%M --strategy selective_20 --train_split train_100 --epochs 30 --batch_size 32

    echo [%%M] Training: Few-Shot 20%% Data...
    python train.py --model %%M --strategy full --train_split train_20 --epochs 20 --batch_size 32
    
    echo [%%M] Training: Few-Shot 5%% Data...
    python train.py --model %%M --strategy full --train_split train_05 --epochs 20 --batch_size 32

    echo [%%M] Evaluating Linear Probe t-SNE and ConfMats...
    python eval_model.py --run_name %%M_linear_probe_train_100 --model %%M --plot_embeddings --embed_method tsne

    echo [%%M] Extracting Features and Training Probes...
    python probe_features.py --model %%M --batch_size 64
)

echo ========================================
echo Running Robustness Evaluation Scenario 4.4
echo ========================================
python robustness_test.py --run_names resnet50_full_train_100 densenet121_full_train_100 efficientnet_b0_full_train_100 --batch_size 32

echo ========================================
echo All experiments complete! Check the checkpoints directory.
echo ========================================
pause