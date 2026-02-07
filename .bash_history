ls
ls
module list
module avail
sl
dc- 
hwloc-ls -v output.svg
ls
nano hello_world.c
gcc -fopenmp hello_world.c -o app
export OMP_NUM_THREADS=4
./app
./app > output.txt
ls
nano output.txt
nano job.slurm
sbatch job.slurm
exit
gcc -fopenmp hello_world.c -o app
export OMP_NUM_THREADS=4
./app
exit
idev -A ASC23013 -N 1 -n 2
sbatch job.slurm
showq -u duz19
ls -l
cat output.txt
ls
nano job.slurm
idev -A ASC23013 -N 1 -n 2
sbatch job.slurm
showq -u duz19
ls
nano job.slurm
mkdir Lab1
ls
mv job.slurm Lab1/
ls
mv hello_world.c output.svg output.txt Lab1/
ls
cd app
mv app Lab1/
ls
cd Lab1
ls
git add .
git init
ls
git add .
git commit -m "Lab1"
git remote add origin https://git.txstate.edu/duz19/azman_cs7389D.git
ls
git push origin main
git status
git push origin master
git branch -m main
git push origin main
git pull
ls
git push origin main
git pull --rebase origin main
git status
git push origin main
pwd
ls
git add .
git commit -m "output added"
git push origin main
