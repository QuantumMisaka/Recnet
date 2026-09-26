#!/bin/bash
# Launch round2 only if the automatic final C2 gate is still FAIL.
# Submit after the automatic finalizer, not directly after the campaign.

#SBATCH --job-name=recnet-round2
#SBATCH --account=pku-jianghong
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-gpu
#SBATCH --time=12:00:00
#SBATCH --output=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/slurm-recnet-round2-%j.out
#SBATCH --error=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/slurm-recnet-round2-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
FINALIZER_JOB="${FINALIZER_JOB:-1426314}"
GATE="$R/recnet-runs/reports/c2-final-afterany-$FINALIZER_JOB/c2_gate.json"
CASES="$R/recnet-runs/campaign-c2-round2/cases.tsv"
RUN_ID="${RUN_ID:-campaign-c2-round2-$SLURM_JOB_ID}"
CONCURRENCY="${CONCURRENCY:-4}"
THREADS="${THREADS:-4}"

[[ -f "$GATE" ]] || { echo "[round2] final gate missing: $GATE" >&2; exit 2; }
[[ -f "$CASES" ]] || { echo "[round2] round2 cases missing: $CASES" >&2; exit 2; }

source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$R/Recnet${PYTHONPATH:+:$PYTHONPATH}"

if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("gate_pass") is True else 1)' "$GATE"; then
    echo "[round2] final C2 gate PASS; no round2 launch"
    exit 0
fi

AUDIT="$(dirname "$GATE")/missing_attempt_audit.json"
if [ -f "$AUDIT" ]; then
    if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); xs=d.get("items",[]); sys.exit(0 if xs and all(x.get("status")=="attempted_rejected" for x in xs) else 1)' "$AUDIT"; then
        echo "[round2] all missing channels were already attempted and rejected by first-round free search"
        echo "[round2] skip blind duplicate run; route to boundary/D2S strategy review"
        exit 0
    fi
fi

RUN_DIR="$R/recnet-runs/$RUN_ID"
mkdir -p "$RUN_DIR"

if [ -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]; then
    source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "$CUDA_VISIBLE_DEVICES" ]] || { echo "[round2] CUDA allocation unknown" >&2; exit 4; }

echo "[round2] final gate FAIL; launching targeted round2"
echo "[round2] gate=$GATE cases=$CASES run_dir=$RUN_DIR"
echo "[round2] devices=$CUDA_VISIBLE_DEVICES concurrency=$CONCURRENCY threads=$THREADS"

python "$R/Recnet/recnet_adapter/tools/farm_runner.py" \
  --cases "$CASES" \
  --concurrency "$CONCURRENCY" \
  --threads "$THREADS" \
  --recnet-dir "$R/Recnet" \
  --run-id "$RUN_ID" \
  --summary "$RUN_DIR/farm_summary.tsv"

echo "[round2] farm complete; regenerate final C2 gate before C3 decision"
