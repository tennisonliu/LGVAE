#!/bin/bash 
#SBATCH -J Sim2ProVAE # Job name 
#SBATCH -n 16 # Number of total cores 
#SBATCH -N 1 # Number of nodes 
#SBATCH -A venkvis_gpu
#SBATCH --gres=gpu:1
#SBATCH -p gpu 
#SBATCH --mem-per-cpu=2000 # Memory pool for all cores in MB 
#SBATCH --time=14-00:00:00
#SBATCH -e error_%j.err 
#SBATCH -o out_%j.out # File to which STDOUT will be written %j is the job # 
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=mingzeya@andrew.cmu.edu # Email to which notifications will be sent 

echo "Job started on `hostname` at `date`" 
CRTDIR=$(pwd) # get current work directory
echo $CRTDIR
cd $CRTDIR

source /home/mingzeya/miniconda3.sh 

conda activate abnn

module load cuda/10.2.89 

cd $CRTDIR

python3 train.py

echo " " 
 echo "Job Ended at `date`" 
