#!/bin/bash
#SBATCH --job-name=rl_train_sample
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mem=64GB 
#SBATCH --time=20:00:00 
#SBATCH --gres=gpu:1

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
export TRANSFORMERS_CACHE=/tmp/$USER


singularity exec --nv --overlay \
$SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate true_few_show
cd /scratch/ra3136/nlu/NLUProject/
nvidia-smi
python3 red_lm/rl_train.py
"

