#!/bin/bash
#SBATCH -J hwlocinfo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:02:00
#SBATCH -p normal
#SBATCH -o hwlocinfo.%j.out

module load hwloc 2>/dev/null || true

echo "==== hostname ===="
hostname

echo "==== lstopo-no-graphics ===="
lstopo-no-graphics

echo "==== lscpu ===="
lscpu

echo "==== free -h ===="
free -h
