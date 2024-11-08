#!/bin/bash
#
#SBATCH --job-name=vlpr
#SBATCH --output=vlpr.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END
#SBATCH --mail-user=zl3958@nyu.edu

module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zl3958/DDPM/;
export PATH=/scratch/zl3958/DDPM/bin:$PATH;
cd /scratch/zl3958/VLPR

python -W ignore AnyLoc/dvgl_benchmark/testing.py --train_batch_size 8 \
	--infer_batch_size 100 --epochs_num 30 --queries_per_epoch 100000 --cache_refresh_rate 100000 --lr 1e-4 \
	--dataset_name pitts30k --datasets_folder /vast/zl3958


