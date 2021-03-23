#!/bin/bash
#SBATCH -p gpu-he --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohit_vaishnav@brown.edu
#SBATCH -n 8
#SBATCH --mem=200G
#SBATCH -N 1
# Specify an output file
#SBATCH -o ../../slurm/fossil/%j.out
#SBATCH -e ../../slurm/fossil/%j.err

# Specify a job name:
#SBATCH --time=100:00:00
module load cuda/10.2
module load gcc/8.3
source ~/data/data/mvaishn1/env/detectron2_fossil/bin/activate
module load python/3.7.4

# export CUDA_VISIBLE_DEVICES=1

python train_net_builtin.py --num-gpus 4 --dist-url tcp://127.0.0.2:1234  --config-file configs/Base_image_resnet_temp.yaml --resume

#sbatch -J fossil scripts/train_net_builtin_he.sh python train_net_builtin.py --num-gpus 1  --config-file configs/Base_image_resnet.yaml --eval-only
