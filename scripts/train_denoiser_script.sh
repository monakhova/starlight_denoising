#!/bin/bash
#SBATCH -p g24 -c 16 
#SBATCH --gres=gpu:8 

set -x

source ../../anaconda3/etc/profile.d/conda.sh
conda activate starlight

python train_denoiser.py --batch_size 1 --gpus 8 --notes test8 --crop_size 256 --data stills_realvideo --noise_type unetfourier --network dvdhr --multiply gamma 