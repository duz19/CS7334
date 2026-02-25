#!/bin/bash
#SBATCH -J lab3_omp
#SBATCH -o output.o%j
#SBATCH -e output.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:01:00
#SBATCH -A ASC23013

module load gcc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

./compute_stddev 10
