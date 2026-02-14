#!/bin/bash
#SBATCH -J dot_prod_omp
#SBATCH -o output.o%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:01:00
#SBATCH -A ASC23013

module load gcc
export OMP_NUM_THREADS=4
./dot_prod
