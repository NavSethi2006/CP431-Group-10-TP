#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=julia_mpi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --mem=0
#SBATCH --output=julia_%j.out
#SBATCH --error=julia_%j.err

module load gcc openmpi

cd $SLURM_SUBMIT_DIR

mkdir -p run_${SLURM_JOB_ID}

srun ./julia_mpi run_${SLURM_JOB_ID}