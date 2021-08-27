#! /bin/bash
#SBATCH --job-name=stitch_decile
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time="96:00:00"
#SBATCH --output='slurm_logs/stitch_decile_%A_%a.out'
#SBATCH --error='slurm_logs/stitch_decile_%A_%a.out'
#SBATCH --partition compute
#SBATCH --array 0-19
#SBATCH --cpus-per-task=10
#SBATCH --mem=10gb
echo "$SLURM_ARRAY_TASK_ID th decile"
conda activate torch_env

python unzip_mris.py -i /work/rhydian/UKBB_Downloads/mris/ \
				    -o /work/rhydian/UKBB_Downloads/mris-processed/\
				    --thread_idx $SLURM_ARRAY_TASK_ID\
				    --num_threads 20
