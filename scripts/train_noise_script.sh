#!/bin/bash
#SBATCH -p gpu -c 16 
#SBATCH --gres=gpu:8 

set -x

source ../../anaconda3/etc/profile.d/conda.sh
conda activate starlight

python train_gan_noisemodel.py --batch_size 1 --gpus 8