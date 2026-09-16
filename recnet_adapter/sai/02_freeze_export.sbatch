#!/bin/bash
# 单头导出（dp --pt freeze --head ft2dp）+ 加载验证。
# 用途：给 Recnet 提供「零代码改动」的交付形态（单头 .pt2，直接 DP(model=...) 即可）。
# 用法：sbatch 02_freeze_export.sbatch [CKPT 路径]
#   可选环境变量：OUT_DIR / RECNET_DP_HEAD / DPEVA_DPA4_ENV_NAME（默认 dpeva-dpa4-320）
# 注意：freeze 的严格 state_dict 加载对构建版本敏感 —— 2026-07 旧 ckpt 在 dev67 上失败、
#       在 GA 3.2.0 上可运行（AOTInductor 编译较慢，需数分钟）；用 GPU 节点导出得到 CUDA 目标件。
#SBATCH --job-name=recnet-freeze
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --time=00:40:00
#SBATCH --output=slurm-recnet-freeze-%j.out
#SBATCH --error=slurm-recnet-freeze-%j.err

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
OUT_DIR="${OUT_DIR:-$R/recnet-runs/_freeze-check}"
CKPT="${1:-$R/v2.2-ft/runs/dpa4_multitask_forgetting_20260711/runs/replay_lr350e-3/models/model.ckpt-100000.pt}"
HEAD="${RECNET_DP_HEAD:-ft2dp}"
TAG="$(basename "$CKPT" .pt)"
OUT="${OUT_DIR}/frozen_${TAG}_${HEAD}.pt2"

export DPEVA_DPA4_ENV_NAME="${DPEVA_DPA4_ENV_NAME:-dpeva-dpa4-320}"
source "$R/dpeva-git/scripts/env/dpeva-dpa4.env"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

python -c 'import torch, deepmd; print("[freeze] deepmd", deepmd.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
echo "[freeze] ckpt=$CKPT head=$HEAD out=$OUT"

time dp --pt freeze -c "$CKPT" --head "$HEAD" -o "$OUT"
ls -lh "$OUT"

python - "$OUT" <<'PY'
import sys

from ase import Atoms
from deepmd.calculator import DP

model = sys.argv[1]
co = Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.13)])
co.set_cell([20.0, 20.0, 20.0])
co.set_pbc([False, False, False])
co.center()

atoms = co.copy()
atoms.calc = DP(model=model)
energy = float(atoms.get_potential_energy())
forces = atoms.get_forces()
print("frozen .pt2 CO E =", round(energy, 4), "| max|F| =", round(float(abs(forces).max()), 4))
print("ref(ckpt head=ft2dp) = -611.9102 -> diff =", round(energy + 611.9102, 6), "eV")
PY

echo "[freeze] done"
