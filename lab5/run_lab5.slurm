#!/bin/bash
#SBATCH -J lab5
#SBATCH -o lab5.o%j
#SBATCH -e lab5.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 00:05:00

module load intel impi

cd $SCRATCH/lab5
ibrun ./lab5
ls -l noncontig.out
