#!/bin/bash

#SBATCH -c 8


module load nvidia/cuda-10.2
python main.py