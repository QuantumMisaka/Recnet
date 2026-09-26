#!/bin/bash
# Recnet case farm ——SAI 单节点 4×V100 模板（一作业多 case，每 case 独占一卡）。
#
# 与 01_run_pipeline.sbatch 的关系：01 是「一作业一 case 一卡」，本脚本把 N 个 case
# 收进一个 allocation，用 case 级并发换调度效率（RecNet 负载是脉冲式 DPA4 推理，
# 单卡跑单 case 时 GPU 大部分时间在等 CCQN/Sella 的 Python 侧）。
#
# 用法（登录节点、干净 shell；SAI 禁 sbatch --export，参数请用位置参数/环境变量）：
#   sbatch 03_case_farm.sbatch <cases.tsv> [run_id]
#   可选环境变量：
#     RECNET_CASES_PY  覆盖 farm_runner.py 路径
#     RECNET_CONCURRENCY  (默认 4)
#     RECNET_THREADS      (默认 4；每个 case 进程的 OMP/MKL/DP 线程预算)
#     RECNET_DEVICES      (默认继承本作业的 CUDA_VISIBLE_DEVICES；显式给出则为逻辑序号或 UUID)
#     RECNET_DP_MODEL / RECNET_DP_HEAD  清单列留空时的兜底
#     RECNET_DEFAULT_TIME_BUDGET_S      单 case 墙钟预算兜底（秒）
#     RECNET_FARM_SUMMARY               farm_summary.tsv 落盘路径
#   清单格式（TSV，`#` 注释与空行允许）：
#     case_name<TAB>case_dir<TAB>model<TAB>head<TAB>extra_args<TAB>time_budget_s
#
# QOS 说明：4 卡作业不能用 *-1o2gpu（每 job 上限 2 卡），按 $sai-user-guide §4 用
# rush-gpu（24h/4~16 卡）或 huge-gpu/flood-gpu；单卡才是 1o2gpu 的世界。
# 资源三参数：--nodes/--ntasks/--gpus-per-node（禁 --mem/--cpus-per-task）。
#SBATCH --job-name=recnet-farm
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-gpu
#SBATCH --time=12:00:00
#SBATCH --output=slurm-recnet-farm-%j.out
#SBATCH --error=slurm-recnet-farm-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
RECNET_DIR="${RECNET_DIR:-$R/Recnet}"
RUN_ROOT="${RUN_ROOT:-$R/recnet-runs}"
CASES="${1:-${RECNET_CASES:-}}"
RUN_ID="${2:-${RECNET_RUN_ID:-farm-$SLURM_JOB_ID}}"
CONCURRENCY="${RECNET_CONCURRENCY:-4}"
THREADS="${RECNET_THREADS:-4}"
MODEL="${RECNET_DP_MODEL:-}"
HEAD="${RECNET_DP_HEAD:-}"
FARM_RUNNER="${RECNET_CASES_PY:-$RECNET_DIR/recnet_adapter/tools/farm_runner.py}"

if [ -z "$CASES" ]; then
    echo "[recnet-farm] ERROR: 未给清单文件。用法：sbatch 03_case_farm.sbatch <cases.tsv> [run_id]" >&2
    exit 3
fi
if [ ! -f "$CASES" ]; then
    echo "[recnet-farm] ERROR: 清单不存在: $CASES" >&2
    exit 3
fi
if [ ! -f "$FARM_RUNNER" ]; then
    echo "[recnet-farm] ERROR: farm_runner.py 不存在: $FARM_RUNNER" >&2
    exit 3
fi

if [ -f "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash" ]; then
    source "/opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
fi

source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"

# TS/CCQN 依赖 conda env 内的 Sella/JAX/NumPy；用户 site 旧包会导致
# Sella 初始化失败（np.asarray copy keyword/JAX 兼容性错误）。
export PYTHONNOUSERSITE=1

# 运行时版本打印 + Sella 初始化门禁（复用 01/00 的做法）：初始化失败就别开始烧卡。
python - <<'PY'
import importlib.metadata as metadata
import jax
import numpy
import sella
from sella import Sella

try:
    sella_version = metadata.version("Sella")
except Exception:
    sella_version = getattr(sella, "__version__", "?")
print(f"[recnet-farm] runtime: sella={sella_version} ({sella.__file__}) "
      f"jax={jax.__version__} numpy={numpy.__version__}")

import ase
import ase.build
import torch

atoms = ase.build.molecule("H2O")
print(f"[recnet-farm] sella-init: OK (ASE {ase.__version__}, torch {torch.__version__}, "
      f"cuda_available={torch.cuda.is_available()})")
PY
echo "[recnet-farm] sella-init check: PASS"

export PYTHONPATH="$RECNET_DIR:${PYTHONPATH:-}"

RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${RECNET_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"

# 作业内必须能证明「哪些卡属于本 allocation」：设备绑定是 fail-closed 的，
# 空掩码一律拒绝，不猜 0..N-1（否则可能绑到别的作业的卡上）。
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[recnet-farm] ERROR: 作业内 CUDA_VISIBLE_DEVICES 为空 —— 无法确定本 allocation 的卡" >&2
    echo "[recnet-farm]        检查 sbatch 是否只用了 --nodes/--ntasks/--gpus-per-node 三参数；" >&2
    echo "[recnet-farm]        或显式给出 RECNET_DEVICES（逻辑序号或 UUID）。" >&2
    exit 4
fi
GPU_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
if [ "$CONCURRENCY" -gt "$GPU_COUNT" ]; then
    echo "[recnet-farm] WARNING: concurrency=$CONCURRENCY > 可见卡数 $GPU_COUNT；并发会压到 $GPU_COUNT" >&2
    echo "[recnet-farm]          注意：卡上并发 >1 会让多个 case 共享同一张卡的显存（V100 32GB）。" >&2
fi

echo "[recnet-farm] job=${SLURM_JOB_ID} node=$(hostname) partition=${SLURM_JOB_PARTITION}"
echo "[recnet-farm] run_dir=$RUN_DIR cases=$CASES run_id=$RUN_ID"
echo "[recnet-farm] code=$(git -C "$RECNET_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[recnet-farm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[recnet-farm] concurrency=$CONCURRENCY threads=$THREADS"
echo "[recnet-farm] model=${MODEL:-<inherit>} head=${HEAD:-<inherit>}"
nvidia-smi -L || true

ARGS=(
    --cases "$CASES"
    --concurrency "$CONCURRENCY"
    --threads "$THREADS"
    --recnet-dir "$RECNET_DIR"
    --run-id "$RUN_ID"
    --summary "${RECNET_FARM_SUMMARY:-$RUN_DIR/farm_summary.tsv}"
)
if [ -n "${RECNET_DEVICES:-}" ]; then
    ARGS+=(--devices "$RECNET_DEVICES")
fi
if [ -n "$MODEL" ]; then
    ARGS+=(--model "$MODEL")
fi
if [ -n "$HEAD" ]; then
    ARGS+=(--head "$HEAD")
fi
if [ -n "${RECNET_DEFAULT_TIME_BUDGET_S:-}" ]; then
    ARGS+=(--default-time-budget-s "$RECNET_DEFAULT_TIME_BUDGET_S")
fi

echo "[recnet-farm] started at $(date '+%F %T')"
python "$FARM_RUNNER" "${ARGS[@]}"
rc=$?
echo "[recnet-farm] farm_runner exit=$rc"
echo "[recnet-farm] manifest 汇总: $RUN_DIR/<case>/RUN_MANIFEST.txt"
echo "[recnet-farm] finished at $(date '+%F %T')"
exit $rc
