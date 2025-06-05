#!/bin/bash
#SBATCH --job-name=test_src2
#SBATCH --output=test_src2.txt
#SBATCH -p gpu_a100
#SBATCH --time=0:50:00
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=j.tang@tue.nl

# Load modules or software if needed
# module load anaconda/2021.11-pth39
#!/bin/bash

# Load the necessary modules
module load 2023
module load Anaconda3/2023.07-2
conda deactivate
conda init bash
source ~/.bashrc
conda env list
conda activate mm_fsl2

nohup  python "/gpfs/home1/jtang1/multimodal_fsl_99/src/inference.py" > train2out