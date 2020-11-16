#!/bin/bash
#SBATCH -J ACSTC_A
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log_att.out
#SBATCH --error=log.err
#SBATCH -t 80:00:00
#SBATCH --gres=gpu:1

module load cuda/10.0
module load anaconda3/5.3.0
source activate torch

python -u acrostic_attack.py
