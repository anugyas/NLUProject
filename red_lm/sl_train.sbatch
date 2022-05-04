#!/bin/bash
#SBATCH --job-name=sl_nlu_gpt2-large_more_epochs
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mem=64GB 
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:1

module purge
singularity exec --nv \
	    --overlay /scratch/as14770/nlu-proj-singularity/nlu_project.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python supervised.py"
