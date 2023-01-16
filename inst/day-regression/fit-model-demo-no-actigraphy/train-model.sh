#!/bin/bash

#SBATCH --job-name=train-no-accel
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=5GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1

# apptainer shell --shell /bin/bash --nv r-4.2.1-torchcuda.sif
cd /home/mjk56/projects/takeda/day-regression/model-no-actigraphy
apptainer exec --nv ../r-4.2.1-torchcuda.sif Rscript train-model.r

