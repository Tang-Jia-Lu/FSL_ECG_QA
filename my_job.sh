#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH -p gpu_h100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=j.tang@tue.nl


# Load modules or software if needed
# module load anaconda/2021.11-pth39
# Load the necessary modules
module load 2023
module load Anaconda3/2023.07-2
conda deactivate
conda init bash
source ~/.bashrc
conda env list
conda activate mm_fsl2
python --version

nohup python /home/jtang1/multimodal_fsl_99/src/extract_model.py > output.txt 
