#!/bin/bash
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=25
#SBATCH -A plgkwantowy3-cpu
#SBATCH -o job_cem_scipy_output.txt
# the above lines are read and interpreted by sbatch, this one and leter are not
# command, this will be executed on a compute node

module load python/3.10.4-gcccore-11.3.0
source venv/bin/activate

# python3 experiments.py
python3 main.py
