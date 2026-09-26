#!/bin/bash
# Recnet per-site TS-seed scan —— one 4xV100 allocation, many scan tasks.
#
# Usage:
#   sbatch 04_seed_scan.sbatch <scan_plan.tsv> [run_id]
# Plan TSV (comments/blank lines allowed):
#   arm<TAB>site<TAB>vg
# Optional environment:
#   RECNET_SCAN_CASE_ROOT   default $R/recnet-runs/pilot; case is CO-${arm}
#   RECNET_SCAN_SPECIES     default CO; case is <SPECIES>-${arm}
#   RECNET_RXN              default 0
#   RECNET_SCAN_CONCURRENCY default 4
#   RECNET_SCAN_THREADS     default 4
#   RECNET_SCAN_TIMEOUT_S   default 3600 per site
#   RECNET_SCAN_DRY_RUN     set 1 to plan only (no DP and no seed files)
#
# Output:
#   $R/recnet-runs/<run_id>/seed_scan_summary.tsv
#   $R/recnet-runs/<run_id>/logs/<arm>-site<site>-vg<vg>.log
#   <case>/rxn/seeds/<rxn_key>_site_<site>[_vg<vg>].xyz

#SBATCH --job-name=recnet-seedscan
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-gpu
#SBATCH --time=12:00:00
#SBATCH --output=slurm-recnet-seedscan-%j.out
#SBATCH --error=slurm-recnet-seedscan-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
RECNET_DIR="${RECNET_DIR:-$R/Recnet}"
RUN_ROOT="${RUN_ROOT:-$R/recnet-runs}"
PLAN="${1:-${RECNET_SCAN_PLAN:-}}"
RUN_ID="${2:-${RECNET_SCAN_RUN_ID:-seed-scan-$SLURM_JOB_ID}}"
CASE_ROOT="${RECNET_SCAN_CASE_ROOT:-$RUN_ROOT/pilot}"
SPECIES="${RECNET_SCAN_SPECIES:-CO}"
RXN="${RECNET_RXN:-0}"
CONCURRENCY="${RECNET_SCAN_CONCURRENCY:-4}"
THREADS="${RECNET_SCAN_THREADS:-4}"
TIMEOUT_S="${RECNET_SCAN_TIMEOUT_S:-3600}"
DRY_RUN="${RECNET_SCAN_DRY_RUN:-0}"
SCANNER="$RECNET_DIR/recnet_adapter/tools/make_seeds_scan.py"

if [ -z "$PLAN" ]; then
    echo "[seed-scan] ERROR: usage: sbatch 04_seed_scan.sbatch <plan.tsv> [run_id]" >&2
    exit 3
fi
for f in "$PLAN" "$SCANNER"; do
    [ -f "$f" ] || { echo "[seed-scan] ERROR: missing $f" >&2; exit 3; }
done

if [ -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]; then
    source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
fi
source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RECNET_DIR:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[seed-scan] ERROR: CUDA_VISIBLE_DEVICES is empty; allocation cannot be proven" >&2
    exit 4
fi
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
if [ "$CONCURRENCY" -gt "${#DEVICES[@]}" ]; then
    echo "[seed-scan] WARNING: concurrency $CONCURRENCY > visible GPUs ${#DEVICES[@]}" >&2
    CONCURRENCY="${#DEVICES[@]}"
fi

RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
WORK_DIR="$RUN_DIR/workers"
mkdir -p "$LOG_DIR" "$WORK_DIR"

MODEL_S="$R/models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-single100k-v20260919/checkpoints/model.ckpt-100000.pt"
MODEL_M="$R/models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-multi200k-v20260920/checkpoints/model.ckpt-200000.pt"
for m in "$MODEL_S" "$MODEL_M"; do
    [ -f "$m" ] || { echo "[seed-scan] ERROR: missing model $m" >&2; exit 3; }
done

echo "[seed-scan] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[seed-scan] plan=$PLAN run_dir=$RUN_DIR case_root=$CASE_ROOT"
echo "[seed-scan] species=$SPECIES rxn=$RXN"
echo "[seed-scan] code=$(git -C "$RECNET_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[seed-scan] devices=${CUDA_VISIBLE_DEVICES} concurrency=$CONCURRENCY threads=$THREADS timeout_s=$TIMEOUT_S dry_run=$DRY_RUN"
python - <<'PY'
import torch
print(f"[seed-scan] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

# Split plan round-robin into per-worker task files.
n=0
while IFS=$'\t' read -r arm site vg extra; do
    [[ "${arm:-}" && "${arm:0:1}" != "#" ]] || continue
    printf '%s\t%s\t%s\t%s\n' "$arm" "$site" "$vg" "${extra:-}" >> "$WORK_DIR/tasks.$((n % CONCURRENCY)).tsv"
    n=$((n + 1))
done < "$PLAN"
if [ "$n" -eq 0 ]; then
    echo "[seed-scan] ERROR: plan has no tasks" >&2
    exit 3
fi
echo "[seed-scan] tasks=$n"

for ((i=0; i<CONCURRENCY; i++)); do
    device="${DEVICES[$i]}"
    tasks="$WORK_DIR/tasks.$i.tsv"
    worker="$WORK_DIR/worker.$i.sh"
    summary="$WORK_DIR/summary.$i.tsv"
    : > "$summary"
	{
        cat <<EOF
#!/bin/bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES="$device"
tasks="$tasks"
summary="$summary"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RECNET_DIR:\${PYTHONPATH:-}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export DP_INTRA_OP_PARALLELISM_THREADS="$THREADS"
export DP_INTER_OP_PARALLELISM_THREADS="$THREADS"
EOF
        cat <<'EOF'
while IFS=$'\t' read -r arm site vg extra; do
    [[ "${arm:-}" ]] || continue
    case "$arm" in
        S) model="$MODEL_S" ;;
        M) model="$MODEL_M" ;;
        *) echo "bad arm $arm" >&2; continue ;;
    esac
case_dir="$CASE_ROOT/$SPECIES-$arm"
    tag="${arm}-site${site}-vg${vg}"
    log="$LOG_DIR/$tag.log"
    seed_glob="$case_dir/rxn/seeds/*_site_${site}_vg${vg}.xyz"
    echo "[$(date '+%F %T')] START $tag device=$CUDA_VISIBLE_DEVICES" >> "$log"
    args=(--rxn "$RXN" --site "$site" --vg "$vg" --model "$model" --head ft2dp
          --bottom-freeze-threshold 13.456)
    if [ "${RECNET_SCAN_DRY_RUN:-0}" = "1" ]; then args+=(--dry-run); fi
    if [ -n "${extra:-}" ]; then read -r -a extra_args <<< "$extra"; args+=("${extra_args[@]}"); fi
    timeout "${RECNET_SCAN_TIMEOUT_S:-3600}" python "$SCANNER" "$case_dir" "${args[@]}" >> "$log" 2>&1
    rc=$?
    seed="$(compgen -G "$seed_glob" | head -1 || true)"
    manifest="$(compgen -G "$case_dir/rxn/seeds/SCAN_MANIFEST_*_site_${site}_vg${vg}.json" | head -1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tag" "$rc" "${seed:-none}" "$manifest" "$log" \
        "$(date '+%F %T')" "$extra" >> "$summary"
    echo "[$(date '+%F %T')] END $tag rc=$rc" >> "$log"
done < "$tasks"
EOF
    } > "$worker"
    chmod +x "$worker"
done

pids=()
for ((i=0; i<CONCURRENCY; i++)); do
    MODEL_S="$MODEL_S" MODEL_M="$MODEL_M" CASE_ROOT="$CASE_ROOT" RXN="$RXN" \
    SPECIES="$SPECIES" \
    LOG_DIR="$LOG_DIR" SCANNER="$SCANNER" RECNET_SCAN_TIMEOUT_S="$TIMEOUT_S" \
    RECNET_SCAN_DRY_RUN="$DRY_RUN" \
        bash "$WORK_DIR/worker.$i.sh" &
    pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

{
    printf 'tag\trc\tseed\tmanifest\tlog\tended_at\textra_args\n'
    cat "$WORK_DIR"/summary.*.tsv
} > "$RUN_DIR/seed_scan_summary.tsv"
echo "[seed-scan] summary=$RUN_DIR/seed_scan_summary.tsv"
cat "$RUN_DIR/seed_scan_summary.tsv"
echo "[seed-scan] finished at $(date '+%F %T') rc=$rc"
exit "$rc"
