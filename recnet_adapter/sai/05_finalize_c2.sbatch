#!/bin/bash
# Automatic C2 finalization after a campaign farm reaches any terminal state.

#SBATCH --job-name=recnet-c2-final
#SBATCH --account=pku-jianghong
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=rush-cpu
#SBATCH --time=04:00:00
#SBATCH --output=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/slurm-recnet-c2-final-%j.out
#SBATCH --error=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/slurm-recnet-c2-final-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
FINALIZER="$R/Recnet/recnet_adapter/tools/finalize_network.sh"
REPORT="${C2_REPORT:-$R/recnet-runs/reports/c2-final-afterany-$SLURM_JOB_ID}"

[[ -x "$FINALIZER" ]] || { echo "missing finalizer: $FINALIZER" >&2; exit 2; }
mkdir -p "$REPORT"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

"$FINALIZER" \
  --root "$R/recnet-runs/pilot" \
  --root "$R/recnet-runs/co-seed-recovery-v1" \
  --root "$R/recnet-runs/campaign-c2" \
  --expected-root "$R/recnet-runs/campaign-c2" \
  --report "$REPORT" \
  --anchors "$R/validation-pipeline/reaction/paper_network_table.json" \
  --coverage-threshold 0.95

sha256sum "$R/Recnet/recnet_adapter/tools/c2_gate.py" \
          "$R/Recnet/recnet_adapter/tools/finalize_network.sh" \
  | sort -k2 > "$REPORT/TOOL_SHA256SUMS"
sha256sum -c "$REPORT/TOOL_SHA256SUMS"

echo "[c2-final] report=$REPORT"
echo "[c2-final] gate=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate_pass"])' "$REPORT/c2_gate.json")"
