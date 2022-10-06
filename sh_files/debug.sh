#!/bin/sh
#SBATCH --job-name jobname
#SBATCH --error jobname-error.e%j
#SBATCH --output jobname-out.o%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition debug-gpu
#SBATCH --time 00:14:00

srun singularity exec --nv pytorch_geo.sif python3 ~/CGAN-graph-generic/main.py --data_path=/home/users/b/boget3/data/

