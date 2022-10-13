#!/usr/bin/env bash
# Simple array job sample

# Set SLURM options
#SBATCH --job-name=array_factor                 # Job name
#SBATCH --output=array_factor-%A-%a.out        # Standard output and error log
#SBATCH --mail-user=username@middlebury.edu     # Where to send mail    
#SBATCH --mail-type=NONE                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --cpus-per-task=1                       # Run each array job on a single core
#SBATCH --mem=2gb                               # Job memory request
#SBATCH --partition=standard                    # Partition (queue) 
#SBATCH --time=01:00:00                         # Time limit hrs:min:sec
#SBATCH --array=0-7                             # Array range: stets number of array jobs

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Starting: "`date +"%D %T"`

# Your calculations here

python generate_data_with_slurm_driver.py <<< "${SLURM_ARRAY_TASK_ID}"

# End of job info
echo "Ending:   "`date +"%D %T"`
