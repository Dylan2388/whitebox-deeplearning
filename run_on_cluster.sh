#!/bin/bash

#SBATCH -c 8
#SBATCH --constraint="avx2"

module load nvidia/cuda-10.2
python main.py