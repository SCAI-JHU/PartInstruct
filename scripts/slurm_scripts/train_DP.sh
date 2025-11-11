#!/bin/bash
#SBATCH --job-name=train_dp3
#SBATCH --output=/logs/%x_%j.out
#SBATCH --error=/logs/%x_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --nodelist=h100

# Activate conda environment (edit path if needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate partinstruct

MASTER_ADDR=localhost
MASTER_PORT=23456  # Any free port

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

nvidia-smi

torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=manual_run \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    PartInstruct/baselines/training/run_training.py \
    --config-name=DP-S \
    run_name="Train_DP3-$(date +'%m-%d_%H-%M-%S')" \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    task_name=kitchenpot \
    training.resume.resume=False \
    training.resume.epoch=2800 \
    training.resume.path='your_path' \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    training.checkpoint_every=100 \
    dataset.dataset_path=PartInstruct/data/demos/scissors.hdf5

conda deactivate