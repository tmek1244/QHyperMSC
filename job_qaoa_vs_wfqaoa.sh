#!/bin/bash
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH -A plgkwantowy3-cpu
#SBATCH -o job_output.txt
# the above lines are read and interpreted by sbatch, this one and leter are not
# command, this will be executed on a compute node

module load python/3.10.4-gcccore-11.3.0
source venv/bin/activate

# python3 experiments.py
python3 qaoa_vs_wfqaoa.py kp_1 qaoa
python3 qaoa_vs_wfqaoa.py kp_1 wfqaoa
python3 qaoa_vs_wfqaoa.py kp_2 qaoa
python3 qaoa_vs_wfqaoa.py kp_2 wfqaoa
python3 qaoa_vs_wfqaoa.py kp_3 qaoa
python3 qaoa_vs_wfqaoa.py kp_3 wfqaoa
