#!/bin/bash
#SBATCH -A ASC23013
#SBATCH -J pi_omp
#SBATCH -o pi_omp.%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -p normal
#SBATCH -t 00:05:00

module reset
module load gcc   # only if needed

# Use all cores on the node
export OMP_NUM_THREADS=128

# Example placement policy (change per experiment)
echo "===== core-close ====="
export OMP_PLACES=cores
export OMP_PROC_BIND=close
./compute_pi 10000000

echo "===== core-spread ====="
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
./compute_pi 10000000

echo "===== socket-close ====="
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
./compute_pi 10000000

echo "===== socket-spread ====="
export OMP_PLACES=sockets
export OMP_PROC_BIND=spread
./compute_pi 10000000
