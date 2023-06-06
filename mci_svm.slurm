#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name="mci_svm"
#SBATCH --mem=5GB
#SBATCH --time=15:00:00
#SBATCH --output=mci_svm.outs
#SBATCH --mail-user=musto101@googlemail.com
#SBATCH --mail-type=ALL

cd /my_igor_work/deterioration_nested_bootstrap

srun -l python  MCI/mci_svm.py


