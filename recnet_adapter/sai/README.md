# SAI 端运行说明（FT2DP/DPA4 后端）

> 对应工作流事实源：`../../../../docs/tasks/reaction-network-dpa4.md`（§SAI 就绪）。
> 平台规则遵循 `$sai-user-guide`：单卡作业用 `4V100` + `rush-1o2gpu`/`flood-1o2gpu`；
> 只给 `--nodes/--ntasks/--gpus-per-node` 三参数（禁 `--mem`/`--cpus-per-task`）；提交 shell 保持干净。

## 0. 前置（一次性）

```bash
# 登录节点：同步代码（本地 -> SAI）
rsync -av --exclude '__pycache__' --exclude '*.pyc' \
  /home/james/work/ft2dp-dpeva/Recnet/ \
  SAI-new:/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/Recnet/

# 环境：**复用 SAI `dpeva-dpa4`，无需新建**（deepmd 3.2.0b1.dev67 / torch 2.11+cu126 /
# ase / sella / pyyaml / scipy / vesin，实测运行链路 import 全通过）；GA 的 `dpeva-dpa4-320` 为备用。
# rdkit+molecule 只在 RMG 数据准备步需要，不进管线（决策与备选见 ../NOTES-deepmd-compat.md §7）。
```

模型（已在集群，无需上传；SHA256 与本地已验证件一致）：

| 用途 | 路径 |
|---|---|
| 多头 FT2DP（`Default`+`ft2dp`，推荐） | `$R/v2.2-ft/runs/dpa4_multitask_forgetting_20260711/runs/replay_lr350e-3/models/model.ckpt-100000.pt`（SHA `6152d7b2…`） |
| 单头 FT2DP（`head=none`） | `$R/v2.2-ft/runs/dpa4_air_matpes_zbl_train_data_iter11_50k/resume_from_30000_nofullval_numworkers0/models/model.ckpt-50000.pt` |
| 基座（未微调对照） | `$R/models/426/DPA4-MatPES-ZBL-v20260629/checkpoints/DPA4-Air-ZBL-MatPES-v20260629.pt` |

## 1. 后端冒烟（不需要任何化学输入）

```bash
ssh SAI-new
cd /org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/Recnet/recnet_adapter/sai
sbatch 00_backend_check.sbatch
# 结果：$R/recnet-runs/backend-check-<时间戳>/slurm-recnet-check-<jobid>.out
# 期望：result: PASS（含 head 注入对比、HarmonicallyForcedDP、E+F 时延）
```

## 2. 管线运行（需要 case 输入）

准备 `CASE` 目录（建议放 `$R/recnet-runs/<case>/`）：

```
<case>/
├── <slab>.cif                 # 例：surf_07_Fe2C(110)-3x2.cif
└── prepared_data/
    └── prepared_rmg_data.yaml # 由 prepare_rmg_data.py 从 RMG 反应 yaml 生成
```

```bash
RECNET_CASE=$R/recnet-runs/<case> \
RECNET_SLAB=$R/recnet-runs/<case>/surf.cif \
RECNET_EXTRA_ARGS="--use-c-vacancy-io --gas-species-whitelist sp_002" \
  sbatch 01_run_pipeline.sbatch
```

输出：`<case>/rxn/{Adsorbates,TS_guesses,FS_energy,...}` + `slurm-recnet-rxn-<jobid>.{out,err}`。

## 3. 代码同步流程（本地 → fork → SAI）

适配代码的规范位置 = `QuantumMisaka/Recnet` 的 `feat/ft2dp-dpa4-adapter` 分支
（`origin`=你的 fork 可推送；`upstream`=师弟仓库，只读）。

```bash
# ① 本地：跟随上游最新（有冲突解决后再继续）
git -C ~/work/ft2dp-dpeva/Recnet fetch upstream --prune
git -C ~/work/ft2dp-dpeva/Recnet rebase upstream/main

# ② 本地：提交并推送分支到 fork
git -C ~/work/ft2dp-dpeva/Recnet push origin feat/ft2dp-dpa4-adapter

# ③ 同步到 SAI（含 .git，作业日志会打印部署 commit）
rsync -av --exclude '__pycache__' --exclude '*.pyc' \
  ~/work/ft2dp-dpeva/Recnet/ \
  SAI-new:/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/Recnet/

# ④ 三端核对（应一致）
ssh SAI-new 'cd /org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/Recnet && git log --oneline -1'
```

约定：作业产物（`slurm-*.out/.err`、计算输出）留在 `$R/recnet-runs/<case>/`，不落代码目录。

## 4. 备注

- 单卡足够：Recnet 是单进程 ASE + `DP` 推理（deepmd pt 单进程单卡）；多卡不加速本流程。
- **单头导出（可选交付形态）**：`02_freeze_export.sbatch` 用 `dp --pt freeze --head ft2dp` 生成单头 `.pt2`，
  这样对方 `DP(model=...)` 不带 head 也能用（零代码改动）。注意：导出用 GA 构建（`dpeva-dpa4-320`），
  AOTInductor 编译需数分钟；导出后在作业内自动做 CO 单点验证（参考值 −611.9102 eV）。
- `prepare_rmg_data.py`（RMG yaml → prepared）需要 `rdkit` + `molecule`：
  建议在本地/师弟侧生成后随 case 同步，或在 SAI 侧 `pip install rdkit rmg-molecule`（需先确认镜像源）。
- 每个进程入口都需经过 `recnet_adapter.install()`（用 `python -m recnet_adapter` 或
  `run_dp_ts_ft2dp.py` 启动即可）；直接 `python run_dp_ts.py` 不会应用映射。
