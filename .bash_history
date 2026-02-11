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
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
sacct -j 2913670 --format=JobID,State,ExitCode,Elapsed
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
squeue -u duz19
NOTIFICATION: sbatch not available on compute nodes. Use a login node.
ls
cd Lab2
ls
sbatch batch_submission.sh 
squeue -u duz19
squeue -u duz19
ls
nano batch_submission.sh 
sbatch batch_submission.sh 
squeue -u duz19
squeue -u duz19
ls
hostname
ls
nano batch_submission.sh 
sbatch batch_submission.sh 
ls
squeue -u duz19
nano batch_submission.sh 
sbatch batch_submission.sh 
ls
sacct -j 2914005 --format=JobID,State,ExitCode,Elapsed
sacct -j 2913692 --format=JobID,State,ExitCode,Elapsed
ls
nano lab2ring.2913692.out 
nano batch_submission.sh 
sbatch batch_submission.sh 
ls
nano lab2ring.2913692.out 
nano lab2ring.2913693.out
nano lab2ring.2913693.out
ls
rm prog1 prog2
ls
cd ..
ls
cd Lab
cd Lab2
ls
rm lab2ring.2913693.out lab2ring.2913692.out
ls
rm batch_submission.sh 
ls
ls
mkdir Lab2
ls
cd Lab2
ls
touch prog1.c
nano prog1.c
touch prog2.c
nano prog2.c
idev -A ASC23013 -N 1 -n 4
idev -A ASC23013 -N 1 -n 3
idev -A ASC23013 -N 1 -n 4
idev -A ASC23013 -N 1 -n 4 -p development -t 00:10:00
idev -A ASC23013 -N 1 -n 4 -p development -t 00:30:00
idev -A ASC23013 -N 1 -n 4 -p development -t 00:30:00
idev -A ASC23013 -N 1 -n 4 -p development -t 00:10:00
idev -A ASC23013 -N 1 -n 4 -p normal -t 00:30:00
git rm -f --cached Lab1
ls
cd ..
ls
git rm -f --cached Lab1
git rm -f --cached Lab2
rm -rf Lab1/.git
rm -rf Lab1/.git
rm -rf Lab2/.git
git add Lab1 Lab2
git commit -m "Add Lab1 and Lab2 as normal folders"
git pull --rebase origin main
git status
git restore .bash_history
git restore .slurm/idv10394.o2913681
nano .gitignore
git status
git add .gitignore
git commit -m "Ignore slurm and history files"
git add Lab1 Lab2
git commit -m "Add Lab1 and Lab2 submissions"
git pull --rebase origin main
git pull --rebase origin main
git remote -v
git remote add origin https://git.txstate.edu/duz19/azman_cs7389D.git
git fetch origin
git fetch origin
git fetch origin
ls
git branch -r
git pull --rebase origin main
git pull --rebase origin main
git checkout -b main
git pull --rebase origin main
git push -u origin main
hostname
git config --global pack.threads 1
git config --global core.compression 0
git push -u origin main
ls
ls -l
cd app
ls
nano .gitignore
echo ".vscode-server/" >> .gitignore
echo ".vscode/" >> .gitignore
echo "slurm-*.out" >> .gitignore
echo ".slurm/" >> .gitignore
echo ".bash_history" >> .gitignore
nano .gitignore
git push -u origin main
git add .gitignore
git commit -m "Ignore VS Code server and slurm files"
git push --force-with-lease -u origin main
git rev-list --objects --all | grep -F ".vscode-server/cli/servers/" | head
git filter-repo --help >/dev/null 2>&1 && echo "filter-repo exists" || echo "NO filter-repo"
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch .vscode-server" --prune-empty --tag-name-filter cat -- --all
rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git rev-list --objects --all | grep -F ".vscode-server/cli/servers/" | head
printf "\n.vscode-server/\n.vscode/\n.slurm/\nslurm-*.out\n.bash_history\n" >> .gitignore
git add .gitignore
git commit -m "Ignore VS Code server and slurm files" || true
git push --force-with-lease -u origin main
git push --force-with-lease -u origin main
