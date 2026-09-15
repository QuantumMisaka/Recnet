#!/bin/bash
# Recnet 管线（FT2DP/DPA4 后端）——SAI 单卡 V100 模板。
#
# 前置：CASE 目录内已备好 slab（cif/poscar）与 prepared_data/prepared_rmg_data.yaml
#（RMG 数据准备步骤见 docs/tasks/reaction-network-dpa4.md；本脚本不执行 prepare_rmg_data.py）。
#
# 用法（登录节点、干净 shell）：
#   RECNET_CASE=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/<case> \
#     sbatch 01_run_pipeline.sbatch [--top-x 1 ...]
# 可选覆盖：RECNET_DIR / RECNET_SLAB / RECNET_PREPARED / RECNET_DP_MODEL / RECNET_DP_HEAD
#           RECNET_EXTRA_ARGS（如需 --use-c-vacancy-io 等附加参数）
# 说明：qos 默认 flood-1o2gpu（低优，最长 4h）；长任务改 rush-1o2gpu（24h）并同步调 --time。
#SBATCH --job-name=recnet-rxn
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=04:00:00
#SBATCH --output=slurm-recnet-rxn-%j.out
#SBATCH --error=slurm-recnet-rxn-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
RECNET_DIR="${RECNET_DIR:-$R/Recnet}"
CASE="${RECNET_CASE:?请用 RECNET_CASE=<case 目录> 提交，例如 $R/recnet-runs/fe2c110-demo}"
SLAB="${RECNET_SLAB:-}"
PREPARED="${RECNET_PREPARED:-$CASE/prepared_data/prepared_rmg_data.yaml}"
MODEL="${RECNET_DP_MODEL:-$R/v2.2-ft/runs/dpa4_multitask_forgetting_20260711/runs/replay_lr350e-3/models/model.ckpt-100000.pt}"
HEAD="${RECNET_DP_HEAD:-ft2dp}"

if [ ! -f "$PREPARED" ]; then
    echo "[recnet-sai] ERROR: prepared 数据不存在: $PREPARED" >&2
    echo "[recnet-sai] 提示：先把 RMG 反应 yaml 转成 prepared_rmg_data.yaml（需 rdkit/molecule），" >&2
    echo "[recnet-sai]       或从师弟工作目录同步 prepared_data/ 后再提交。" >&2
    exit 3
fi

if [ -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]; then
    source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
fi

source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"

export RECNET_DP_MODEL="$MODEL"
export RECNET_DP_HEAD="$HEAD"
export PYTHONPATH="$RECNET_DIR:${PYTHONPATH:-}"

cd "$CASE"
echo "[recnet-sai] job=${SLURM_JOB_ID} node=$(hostname) case=$CASE"
echo "[recnet-sai] code=$(git -C "$RECNET_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[recnet-sai] model=$MODEL head=$HEAD"
nvidia-smi -L || true

ARGS=(--path "$CASE" --prepared "$PREPARED" --top-x 1)
if [ -n "$SLAB" ]; then
    ARGS+=(--slab "$SLAB")
fi
if [ -n "${RECNET_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=(${RECNET_EXTRA_ARGS})
fi

python "$RECNET_DIR/recnet_adapter/run_dp_ts_ft2dp.py" "${ARGS[@]}" "$@"

echo "[recnet-sai] finished at $(date '+%F %T')"
