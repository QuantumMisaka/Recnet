#!/bin/bash
# RecNet 启动方案支撑实验：一张 4V100 卡上「几个 case 进程」的稳态吞吐。
#
# 结论用途：决定 03_case_farm.sbatch 的 --gpus-per-node / RECNET_CONCURRENCY 配比
# （当前 1 case/卡、concurrency=4）。探针只读模型与结构，不写 case 目录。
#
# 平台规则遵循 $sai-user-guide：单卡作业 4V100 + rush-1o2gpu；只用
# --nodes/--ntasks/--gpus-per-node 三参数。
#SBATCH --job-name=recnet-gpu-probe
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --time=00:40:00
#SBATCH --output=slurm-recnet-gpu-probe-%j.out
#SBATCH --error=slurm-recnet-gpu-probe-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
RECNET_DIR="${RECNET_DIR:-$R/Recnet}"
RUN_ROOT="${RUN_ROOT:-$R/recnet-runs}"
MODEL="${RECNET_DP_MODEL:-$R/models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-single100k-v20260919/checkpoints/model.ckpt-100000.pt}"
HEAD="${RECNET_DP_HEAD:-ft2dp}"
TS_FRAME="${RECNET_TS_FRAME:-$R/recnet-runs/campaign-c2/S-c0/rxn/TS_guesses/rxn_083110bcd934_site_1_vg1.xyz}"
MD_FRAME="${RECNET_MD_FRAME:-$R/validation-pipeline/md_supercells/runs_multi200k/surface_Fe5C2chi510_fe5c2kp0009_x2x1x1__nvt__600K/POSCAR}"
WORKERS="${RECNET_PROBE_WORKERS:-1,2,4,8}"
REPEATS="${RECNET_PROBE_REPEATS:-300}"

if [ -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]; then
    source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
fi

source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RECNET_DIR:${PYTHONPATH:-}"

RUN_DIR="$RUN_ROOT/gpu-probe-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[gpu-probe] ERROR: CUDA_VISIBLE_DEVICES 为空，无法确定本 allocation 的卡" >&2
    exit 4
fi

echo "[gpu-probe] job=${SLURM_JOB_ID} node=$(hostname) partition=${SLURM_JOB_PARTITION}"
echo "[gpu-probe] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[gpu-probe] model=${MODEL} head=${HEAD} workers=${WORKERS} repeats=${REPEATS}"
echo "[gpu-probe] run_dir=$RUN_DIR"

python - <<'PY'
import importlib.metadata as metadata
import torch, ase, deepmd
print(f"[gpu-probe] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"ase={ase.__version__} deepmd={metadata.version('deepmd-kit')}")
if torch.cuda.is_available():
    print(f"[gpu-probe] device0={torch.cuda.get_device_name(0)}")
PY

PROBE="$RECNET_DIR/recnet_adapter/tools/gpu_concurrency_probe.py"

echo "=== A. TS 尺寸体系（143 原子，RecNet 鞍点搜索代表）==="
python "$PROBE" --structure "$TS_FRAME" --model "$MODEL" --head "$HEAD" \
    --workers "$WORKERS" --repeats "$REPEATS" --out "$RUN_DIR/probe_ts143.tsv" \
    2>&1 | tee "$RUN_DIR/probe_ts143.log"

echo "=== B. MD 超胞（242 原子，真实工况代表）==="
python "$PROBE" --structure "$MD_FRAME" --model "$MODEL" --head "$HEAD" \
    --workers "$WORKERS" --repeats "$REPEATS" --out "$RUN_DIR/probe_md242.tsv" \
    2>&1 | tee "$RUN_DIR/probe_md242.log"

echo "[gpu-probe] done; results in $RUN_DIR"
