#!/bin/bash
#SBATCH -A ASC23013
#SBATCH -J pi_sched
#SBATCH -o pi_sched.%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -p normal
#SBATCH -t 00:10:00

module reset

export OMP_PLACES=sockets
export OMP_PROC_BIND=spread

export OMP_NUM_THREADS=128
N=10000000

echo "N=$N"
echo "Policy,Chunk,Time(s)"

for policy in static dynamic; do
  for chunk in 10 100 1000; do
    export OMP_SCHEDULE="${policy},${chunk}"
    out=$(./compute_pi $N | grep -E "Time")
    time=$(echo "$out" | awk '{print $3}')
    echo "${policy},${chunk},${time}"
  done
done
