#!/bin/bash
#SBATCH -J lab1_omp
#SBATCH -o output.txt
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:01:00
#SBATCH -A ASC23013

module load gcc
export OMP_NUM_THREADS=4
./app

