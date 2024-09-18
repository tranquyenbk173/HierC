#!/bin/bash -e

#SBATCH --job-name=hierC # create a short name for your job
#SBATCH --output=/home/quyentt15/quyentt15/sbatch_out/mbpp%A.out # create a output file
#SBATCH --error=/home/quyentt15/quyentt15/sbatch_err/mbpp%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.quyentt15@vinai.io

# Activate the conda environment
source ~/quyentt15/envs/anaconda3/bin/activate hide

cd /home/quyentt15/quyentt15/Hierachical_CL/

sh training_scripts/train_cifar100_vit.sh