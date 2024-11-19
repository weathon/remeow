#!/bin/bash

#SBATCH --time=24:00:00               # Walltime
#SBATCH --nodes=1                     # Number of Nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --gres=gpu:1
#SBATCH --mem=12G                     # Memory total in GB (for all cores)
#SBATCH --job-name=baseline     # Job name
#SBATCH --account=st-dushan20-1-gpu   # Account name
#SBATCH --mail-user=wg25r@student.ubc.ca   # Where to send mail
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --output=output.txt  # Standard output
#SBATCH --error=error.txt   # Standard error

# Clear the error file
> error.txt

# Load user's bash profile
source /home/wg25r/.bashrc

# Set environment variables
export WANDB_CONFIG_DIR="/scratch/st-dushan20-1/"
export TORCH_HOME="/scratch/st-dushan20-1/" 

# Load modules
module load http_proxy
module load miniconda3

# Change to the working directory
#cd /home/wg25r/with_pretrain

cd /arc/burst/st-dushan20-1/meow/BSUV-Net-2.0
# Run the Python script
python3 main.py --set_number 1
