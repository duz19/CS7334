#!/bin/bash
#SBATCH -A ASC23013
#SBATCH -J pi_scale
#SBATCH -o pi_scale.%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -p normal
#SBATCH -t 00:10:00

module reset

export OMP_PLACES=sockets
export OMP_PROC_BIND=spread
export OMP_SCHEDULE="static"     # keep constant for scaling

N=10000000
echo "N=$N"
echo "Threads,Time(s)"

for t in 1 2 4 8 16 32 64 128; do
  export OMP_NUM_THREADS=$t
  # capture the Time line your program prints
  out=$(./compute_pi $N | grep -E "Time")
  time=$(echo "$out" | awk '{print $3}')
  echo "$t,$time"
done
