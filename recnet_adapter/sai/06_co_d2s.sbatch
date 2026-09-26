#!/bin/bash
# CO dissociation D2S recovery from Recnet overstretch endpoints.

#SBATCH --job-name=recnet-co-d2s
#SBATCH --account=pku-jianghong
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --qos=rush-1o2gpu
#SBATCH --time=12:00:00
#SBATCH --output=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/co-d2s-v1/slurm-recnet-co-d2s-%j.out
#SBATCH --error=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/co-d2s-v1/slurm-recnet-co-d2s-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
CASES="$R/recnet-runs/co-d2s-v1/cases.tsv"
REACTION="$R/validation-pipeline/reaction"

source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"
export PYTHONNOUSERSITE=1
PY="$(command -v python)"
export OMP_NUM_THREADS=4
cd "$REACTION"

S_MODEL="$R/models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-single100k-v20260919/checkpoints/model.ckpt-100000.pt"
M_MODEL="$R/models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-multi200k-v20260920/checkpoints/model.ckpt-200000.pt"
CONC="${CONC:-2}"
mkdir -p "$R/recnet-runs/co-d2s-v1/d2s"

CASE_LIST=()
while IFS=$'\t' read -r name case model head outdir; do
    [[ "${name:-}" ]] || continue
    [[ "${name:0:1}" != "#" ]] || continue
    CASE_LIST+=("$case $model $head $outdir $name")
done < "$CASES"

echo "[co-d2s] job=${SLURM_JOB_ID} cases=${#CASE_LIST[@]} conc=$CONC"
i=0
for spec in "${CASE_LIST[@]}"; do
    read -r case model head outdir name <<<"$spec"
    gpu=$((i % 2))
    echo "[co-d2s] START $name gpu=$gpu"
    (
        CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$REACTION/d2s_run.py" \
          "$case" "$model" "$head" "$outdir" \
          > "$R/recnet-runs/co-d2s-v1/${name}.run.log" 2>&1
    ) &
    i=$((i + 1))
    if (( i % CONC == 0 )); then wait; fi
done
wait

echo "[co-d2s] metas"
for m in "$R/recnet-runs/co-d2s-v1/d2s"/*/meta.json; do
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("model_barrier"), d.get("TS_fmax"), d.get("TS_n_imag"))' "$m" || true
done
echo CO_D2S_DONE
