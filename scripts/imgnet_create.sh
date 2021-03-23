#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohit_vaishnav@brown.edu
#SBATCH -n 10
#SBATCH --mem=50G
#SBATCH -N 1
#SBATCH --account=carney-tserre-condo
# Specify an output file
#SBATCH -o ../../slurm/fossil/%j.out
#SBATCH -e ../../slurm/fossil/%j.err

# Specify a job name:
#SBATCH --time=20:00:00
source ~/data/data/mvaishn1/env/detectron2_fossil/bin/activate
module load python/3.7.4
module load cuda/10.2
module load gcc/7.2


python tools/make_imagenet_json.py