#!/bin/bash
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohit_vaishnav@brown.edu
#SBATCH -n 4
#SBATCH --mem=100G
#SBATCH -N 1
# Specify an output file
#SBATCH -o ../../slurm/fossil/%j.out
#SBATCH -e ../../slurm/fossil/%j.err

# Specify a job name:
#SBATCH --time=2:00:00
module load cuda/10.2
module load gcc/8.3
source ~/data/data/mvaishn1/env/detectron2_fossil/bin/activate
module load python/3.7.4

# export CUDA_VISIBLE_DEVICES=1

python train_net_builtin.py --num-gpus 1 --dist-url tcp://127.0.0.2:143334  --config-file configs/Base_image_resnet_torch.yaml --eval-only

#sbatch -J fossil scripts/train_net_builtin_he.sh python train_net_builtin.py --num-gpus 1  --config-file configs/Base_image_resnet.yaml --eval-only
