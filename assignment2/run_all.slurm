#!/bin/bash
#SBATCH -J pc_4
#SBATCH -p development
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 00:15:00
#SBATCH -o pc4_%j.out
#SBATCH -e pc4_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

make all

for run in 1 2 3 4 5; do
    mpirun -np 16 ./workpool 120 > "workpool_np16_run${run}.txt"
done


