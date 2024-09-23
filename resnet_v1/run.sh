#!/bin/bash

#SBATCH --job-name=resnet.1
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/MLproject/resnet_v1
python -u main.py
"

