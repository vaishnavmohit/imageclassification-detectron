#!/bin/bash
#SBATCH -p gpu --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --account=carney-tserre-condo
#SBATCH --mail-user=mohit_vaishnav@brown.edu
#SBATCH -n 10
#SBATCH --mem=200G
#SBATCH -N 1
# Specify an output file
#SBATCH -o ../../slurm/fossil/%j.out
#SBATCH -e ../../slurm/fossil/%j.err

# Specify a job name:
#SBATCH --time=120:00:00
module load cuda/10.2
module load gcc/8.3
source ~/data/data/mvaishn1/env/detectron2_fossil/bin/activate
module load python/3.7.4

# export CUDA_VISIBLE_DEVICES=1

python train_net_builtin.py --num-gpus 4 --dist-url tcp://127.0.0.1:12349  --config-file configs/Base_image_resnet_temp.yaml 
                            # SOLVER.IMS_PER_BATCH 1014 OUTPUT_DIR "../../dump/fossil/imagenet/5mar-nor-condo/"

#sbatch -J fossil scripts/train_net_builtin_2.sh 