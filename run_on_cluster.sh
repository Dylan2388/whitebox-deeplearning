#!/bin/bash
#SBATCH --gres=gpu:1            # use GPU/CUDA to speed up training (make sure your code support this)
#SBATCH -c 8
#SBATCH --constraint="avx2"

module load nvidia/cuda-10.2
python main.py