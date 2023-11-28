#!/bin/bash

#SBATCH -J combo # job name
#SBATCH -o /bsuhome/zacharykeskinen/spicy-analysis/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)- others: bigmem
#SBATCH -t 06:00:00 # run time (hh:mm:ss) - 1 minute in this example.

echo "Starting."
start=0

eval "$(conda shell.bash hook)"
conda activate spicy

echo "Activated environment.."

python -u /bsuhome/zacharykeskinen/spicy-analysis/src/compare/hans/combo.py


t=$SECONDS
printf 'Time taken: %d days, %d minutes\n' "$(( t/86400 ))" "$(( t/60 - 1440*(t/86400) ))"
echo "Done"
