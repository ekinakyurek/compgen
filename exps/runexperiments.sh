#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --qos=high
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=10
#SBATCH --constraint=xeon-g6
#SBATCH --job-name="scan"
#SBATCH --output=scan.out
#SBATCH --error=scan.error
#SBATCH -a 1-8
./runsinglegpu.sh
