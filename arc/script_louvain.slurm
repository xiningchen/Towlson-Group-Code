#!/bin/bash

####### Reserve computing resources #######
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=3G
#SBATCH --partition=cpu2021

# Email settings 
#SBATCH --mail-user=xininche@ucalgary.ca
#SBATCH --mail-type=ALL

# Set where you want the standard our and standard error to go
#SBATCH --output=./Report/M-%x.%j.out
#SBATCH --error=./Report/M-%x.%j.err

# Name of the python function to be called
MAIN="louvain_using_script"

# Define key parameters for the example calculation.
# $SLURM_ARRAY_TASK_ID is the variable containing the value or values passed in by sbatch --array=

S0=$SLURM_ARRAY_TASK_ID #If you specify the inputs directly on the terminal
# S0=1.8 #if you specify it on the script
# Name of results file or directory
RESULTS_FILE=./subject_auditory
DATA_FILE=./raw
ARGS=" $RESULTS_FILE $DATA_FILE $S0"
MAIN_WITH_ARGS=${MAIN}.py${ARGS}

echo "Calling python function: ${MAIN_WITH_ARGS}"

echo "Starting run at $(date)"
echo "Running on compute node $(hostname)"
echo "Running from directory $(pwd)"

# Load modules you require ex. python
module load python/anaconda3-2018.12

# Can now use conda to load enviroments and modules  
echo "Using python version: $(which python)"

#### Actual Job Call ####
python ${MAIN_WITH_ARGS} > ./Out/${MAIN}_Py_${SLURM_JOB_ID}.out 2>&1

echo "Finished run at $(date)"
