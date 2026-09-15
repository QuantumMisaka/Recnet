#!/bin/bash
# SAI 后端冒烟：在计算节点（单卡 V100）运行 recnet_adapter check。
# 提交（登录节点、干净 shell）：  sbatch 00_backend_check.sbatch
# 覆盖项（可选）：RECNET_DIR / RUN_ROOT / RECNET_DP_MODEL / RECNET_DP_HEAD
#SBATCH --job-name=recnet-check
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --time=00:40:00
#SBATCH --output=slurm-recnet-check-%j.out
#SBATCH --error=slurm-recnet-check-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
RECNET_DIR="${RECNET_DIR:-$R/Recnet}"
RUN_ROOT="${RUN_ROOT:-$R/recnet-runs}"
MODEL="${RECNET_DP_MODEL:-$R/v2.2-ft/runs/dpa4_multitask_forgetting_20260711/runs/replay_lr350e-3/models/model.ckpt-100000.pt}"
HEAD="${RECNET_DP_HEAD:-ft2dp}"

if [ -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]; then
    source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
fi

# SAI DPA4 环境（CUDA 12.6.3 + deepmd-kit dev67 + torch cu126；见 $R/dpeva-git/scripts/env/）
source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"

export RECNET_DP_MODEL="$MODEL"
export RECNET_DP_HEAD="$HEAD"
export PYTHONPATH="$RECNET_DIR:${PYTHONPATH:-}"

RUN_DIR="$RUN_ROOT/backend-check-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "[recnet-sai] job=${SLURM_JOB_ID} node=$(hostname) partition=${SLURM_JOB_PARTITION}"
echo "[recnet-sai] model=$MODEL head=$HEAD"
nvidia-smi -L || true

python -m recnet_adapter check --repeats 5
