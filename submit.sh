#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -J fe2c-110-vac_wg
#SBATCH -o recnet.log
#SBATCH -e recnet.err
#SBATCH -p amd

name=$(awk '/^#SBATCH[[:space:]]+-J[[:space:]]+/ {print $3; exit}' "$0")

source /data/softwares/miniconda3/bin/activate

# conda activate fair-chem

# export PYTHONPATH=/data/home/youyinglong/rec_net/pynta/pynta:$PYTHONPATH

touch JobRun.txt
touch /data/home/youyinglong/log/Job.log

# Job state 
echo $SLURM_JOB_ID >> JobRun.txt
echo "Start at $(date)" >> JobRun.txt

echo "START $name $SLURM_JOB_ID at $(date) in $(pwd)" >> /data/home/youyinglong/log/Job.log

#NSIMUL = 4
if [ -f "prepared_data/prepared_rmg_data.yaml" ]; then
    echo "Prepared data already exists. Skipping data preparation step."
else
    echo "Prepared data not found. Running data preparation step."
    conda activate pynta_env
    python prepare_rmg_data.py --rxns ./WG_more.yaml --out ./prepared_data
fi

conda activate deepmd

bottom_freeze_threshold=$(awk 'BEGIN {print 0.24 * 24.1802}')

python run_dp_ts.py \
  --path ./ \
  --prepared ./prepared_data/prepared_rmg_data.yaml \
  --bottom-freeze-threshold "$bottom_freeze_threshold" \
  --slab "./surf_07_Fe2C(110)-3x2.cif" \
  --top-x 1 \
  --use-c-vacancy-io \
  --gas-species-whitelist sp_002 
    # --vacancy-group-index  \


echo "===== Done ! =====" 

# Job State
echo "End at $(date)" >> JobRun.txt

echo "END $name $SLURM_JOB_ID at $(date) in $(pwd)" >> /data/home/youyinglong/log/Job.log