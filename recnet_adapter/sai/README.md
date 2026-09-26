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
- **必须设置 `PYTHONNOUSERSITE=1`**：SAI 用户 site 里的 Sella 2.3.5 + JAX 0.9 会在 NumPy 1.26 上
  使 Sella 初始化失败；conda env 内是 Sella 2.4.2 + JAX 0.10 + NumPy 2.x。`00/01` 脚本已强制，
  `python -m recnet_adapter check` 也会做 Sella 初始化检查。
- **单头导出（可选交付形态）**：`02_freeze_export.sbatch` 用 `dp --pt freeze --head ft2dp` 生成单头 `.pt2`，
  这样对方 `DP(model=...)` 不带 head 也能用（零代码改动）。注意：导出用 GA 构建（`dpeva-dpa4-320`），
  AOTInductor 编译需数分钟；导出后在作业内自动做 CO 单点验证（参考值 −611.9102 eV）。
- `prepare_rmg_data.py`（RMG yaml → prepared）需要 `rdkit` + `molecule`：
  建议在本地/师弟侧生成后随 case 同步，或在 SAI 侧 `pip install rdkit rmg-molecule`（需先确认镜像源）。
- 每个进程入口都需经过 `recnet_adapter.install()`（用 `python -m recnet_adapter` 或
  `run_dp_ts_ft2dp.py` 启动即可）；直接 `python run_dp_ts.py` 不会应用映射。

---

## 5. Case farm：一个作业跑多个 case（`03_case_farm.sbatch` + `../tools/farm_runner.py`）

> 本节为追加内容（2026-09-21），不改动上面 0–4 节。

### 5.1 何时用它

`01_run_pipeline.sbatch` 是「一作业一 case 一卡」。当 campaign 有 O(100) 个 case 时，
逐个投小作业的问题不在 GPU 算力，而在**排队**：4V100 分区里单卡作业可用的
`rush-1o2gpu`/`flood-1o2gpu` 每用户组上有上限，且每个 case 都要重新走
`conda activate` + 模型加载。RecNet 负载是脉冲式的（16 原子 ≈ 20 ms/E+F），单卡单 case
时 GPU 大部分时间在等 CCQN/Sella 的 Python 侧，所以**同一节点上多 case 并发**比
「多作业每作业一卡」更划算。

分界线：case 总数 ≤ 4、或需要单独盯一个 case 时用 `01`；成批 campaign 用 `03`。

### 5.2 清单格式（`cases.tsv`）

TSV，6 列（**必须是 tab 分隔**，`#` 开头为注释行，空行忽略）：

```
case_name	case_dir	model	head	extra_args	time_budget_s
```

| 列 | 必填 | 语义 | 留空时 |
|---|---|---|---|
| `case_name` | 是 | case 标识；**同时是工作目录名**，须唯一、不含 `/` | —（重复会直接报错，避免产物互相覆盖） |
| `case_dir` | 是 | case 目录（含 `prepared_data/prepared_rmg_data.yaml` 与 slab） | — |
| `model` | 否 | 覆盖 `RECNET_DP_MODEL` | 继承 `--model` → `RECNET_DP_MODEL` → `model_config.json` |
| `head` | 否 | 覆盖 `RECNET_DP_HEAD`；`none`/`null` = 单头/冻结模型 | 同上（按 `RECNET_DP_HEAD` → `model_config.json`） |
| `extra_args` | 否 | 追加给 `run_dp_ts.py` 的参数（空格分隔，支持引号） | 只有 farm 补的 `--path`/`--prepared` |
| `time_budget_s` | 否 | 单 case 墙钟预算（秒）；超时整棵进程组被 TERM→KILL | 继承 `--default-time-budget-s`；再空 = 不限 |

示例（两个臂 + 一个带 C 空位的 case）：

```tsv
# Fe5C2(510) C2 以下通道 · S 臂（single100k）
c1_ch3_S	S:/org/.../recnet-runs/fe5c2-510-c1/CH3/	S:/org/.../models/FT2DPv2.2-...-single100k.pt	ft2dp	--use-c-vacancy-io --top-x 2	14400
c1_co_S	S:/org/.../recnet-runs/fe5c2-510-c1/CO/	S:/org/.../models/FT2DPv2.2-...-single100k.pt	ft2dp	--use-c-vacancy-io --top-x 2	14400
```

`extra_args` 里的 `--env K=V` 是**farm 自己消费的**（case 级环境覆盖，不透传给管线），
用于「只有一个 case 要换 `RECNET_ADAPTER_DEBUG`」这类场景。

### 5.3 提交示例

```bash
ssh SAI-new
cd /org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/Recnet/recnet_adapter/sai

# ① 先本地/CPU 干跑验清单（不建 DP 计算器，秒级）
python ../tools/farm_runner.py --cases $R/recnet-runs/fe5c2-510-campaign/cases.tsv \
    --concurrency 2 --threads 4 --dry-run

# ② 正式提交（SAI 禁 --export：参数走位置参数/环境变量）
RECNET_CONCURRENCY=4 RECNET_THREADS=4 \
  sbatch 03_case_farm.sbatch $R/recnet-runs/fe5c2-510-campaign/cases.tsv farm-S-arm1
```

产物（`RUN_ROOT` 默认 `$R/recnet-runs`）：

```
$R/recnet-runs/<run_id>/slurm-recnet-farm-<jobid>.{out,err}
$R/recnet-runs/<run_id>/farm_summary.tsv              # 汇总（每 case 一行 + totals）
<case_dir>/farm/runs/<run_id>/<case_name>/case.log              # 该 case 的完整 stdout/stderr
<case_dir>/farm/runs/<run_id>/<case_name>/RUN_MANIFEST.txt      # 追加式记录（见 5.5）
<case_dir>/rxn/...                                              # 管线科学产物（与 01 同口径）
```

`run_id` 默认 `farm-<SLURM_JOB_ID>`，所以**两次提交不会互相覆盖**；同一 `run_id`
重跑会在 `RUN_MANIFEST.txt` 里**追加**一条记录（不覆盖历史）。

### 5.4 字段 ↔ atst runtime 映射

对齐 `atst-tools` 冻结接口（`docs/superpowers/specs/2026-09-21-atst-runtime-interface-design.md`
§2/§4/§6；本仓库不依赖 atst-tools，仅语义对齐）：

| farm 侧 | atst runtime 侧 | 语义对齐要点 |
|---|---|---|
| `--devices 0,1` | `runtime.devices: [0, 1]` | 整数 = **继承可见集合内的 0-based 逻辑序号**（不是宿主卡号）；列表保序、重复拒绝、负数/float 拒绝 |
| `--devices GPU-…` | `runtime.devices: [<uuid>]` | 完整物理 UUID，必须在继承集合内可定位（fail-closed，不字面透传） |
| `--devices ""` | `devices: []` / `ATST_VISIBLE_DEVICES=""` | **显式空请求一律拒绝**，不静默退化成 inherit |
| 缺省 `--devices` | `devices` 字段缺省 = inherit | 用 `CUDA_VISIBLE_DEVICES` 的继承集合；未设置时退化为默认 `0..concurrency-1`（并打 WARNING，仅适合已知分到连续卡号的场景） |
| 整机可见 + 显式 `--devices` | allocation unknown + 显式 devices | 拒绝：无法证明分配身份 |
| `CUDA_VISIBLE_DEVICES=2,3` + `--devices 0` | inherited=[2,3] + requested=[0] | 允许（收窄）；子进程实际掩码 = `0`，manifest 记 `device_logical_index: 0` |
| `CUDA_VISIBLE_DEVICES=2,3` + `--devices 2` | 越界 | 拒绝：`device index 2 is outside the inherited visible set (size 2)` |
| `--concurrency N` | `runtime.binding: round_robin`（近义） | 设备按 round-robin 分给并发槽位；**一个 allocation 一个 farm**，不跨节点 |
| `--threads 4` | `runtime.threads: 4` | 在科学库初始化前写入 `OMP/MKL/OPENBLAS/NUMEXPR/DP_INTRA/DP_INTER`；显式 `--threads` 才覆盖调用者已设的值 |
| `RUN_MANIFEST.txt` + `farm_summary.tsv` | `runtime_evidence.json` | 证据载体不同（作业内文本 vs sidecar JSON）：设备/线程/墙钟/失败分类/argv |
| `--dry-run` | —（atst 无对应） | 干跑：目录准备 + prepared 结构体检 + manifest/汇总，**不建 DP 计算器** |

差异说明：atst 的 `runtime` 是**单进程工作流**的启动绑定，farm 是**多 case 编排**，
因此 farm 多出 `--concurrency`、`time_budget_s`、失败分类与汇总表；设备/线程两维
的取值语义与拒绝口径与 atst 保持一致，日后若接 atst 运行时只需替换绑定层。

### 5.5 `RUN_MANIFEST.txt` 记录格式

追加式，每次运行每个 case 追加一段：

```
===== RUN_MANIFEST record <结束时间> =====
record_schema: farm_runner/1
case_name / case_dir / run_id
model / model_sha256          # sha256 流式计算并带 (path,size,mtime) 缓存
recnet_commit                 # git -C <Recnet> rev-parse HEAD，失败=unknown
env_name                      # DPEVA_DPA4_ENV_NAME / CONDA_DEFAULT_ENV
argv / device / device_inherited_visible / device_source / device_logical_index
cuda_visible_devices_effective / threads / thread_env / head / time_budget_s
started_at / ended_at / wall_s / exit_code / failure_class / status / dry_run
work_dir / log / detail
case_rxn_dir / farm_work_dir  # 产物清单（存在时列出）
===== END record =====
```

### 5.6 故障排查

**失败分类**（`failure_class`；顺序 = timeout → import_error → oom → ccqn_unconverged → other）：

| 分类 | 判定证据 | 常见根因与处置 |
|---|---|---|
| `timeout` | 超过 `time_budget_s`，整棵进程组被 TERM→KILL（`exit_code: -15`） | 预算太紧或 case 太大：调 `time_budget_s` / 降 `--top-x`；日志停在哪个 stage 就是超时点 |
| `import_error` | 日志尾部匹配 `ModuleNotFoundError`/`ImportError`/`No module named`/`undefined symbol` | 多半是 `PYTHONNOUSERSITE` 或 env 没生效（section 4 的老问题）；确认走的是 `03` 而不是裸 `python run_dp_ts.py` |
| `oom` | 日志尾部匹配 `CUDA out of memory`/`torch.cuda.OutOfMemoryError`/`std::bad_alloc` 等 | 降 `--concurrency`（多 case 抢同一张卡不会发生，但同卡上别的作业会）、降 `--threads`、或减少单 case 原子数 |
| `ccqn_unconverged` | 日志尾部匹配 `No TS accepted on preferred top-x sites…`、`Optimization failed:`、`did not converge`、`maximum number of steps` | 这是**科学结果**不是脚本 bug：换 `--bottom-freeze-threshold`、加 `--top-x`、或检查 prepared 输入（虚频/端点位点） |
| `other` | 以上都不匹配（或 launcher 自身报错） | 打开 `case.log` 读完整 traceback；`other` 一律保留日志，不做猜测性归因 |

分类是**诊断提示**，不是科学结论；`farm_summary.tsv` 的 `failure_class` 只用于分流复核。

**清单/设备类报错**（退出码 2，直接打 stderr，不会烧卡）：

| 报错 | 原因 | 处置 |
|---|---|---|
| `device index N is outside the inherited visible set (size M)` | `--devices` 用了宿主卡号语义 | 用继承集合内的逻辑序号；先 `echo $CUDA_VISIBLE_DEVICES` 看作业实际拿到什么 |
| `explicit device selection is refused: … not caller-bound …` | 进程外没有 `CUDA_VISIBLE_DEVICES`（例如登录节点手工跑） | 走 `sbatch 03_case_farm.sbatch`，或显式 `CUDA_VISIBLE_DEVICES=0,1 python …` |
| `runtime.devices must not be empty` | `--devices ""` | 要么给具体卡，要么整条参数省略（省略 = inherit） |
| `case_name 重复` | 清单里同名 case | 改名；同名会共用 `farm/runs/<run_id>/<name>/`，产物互相覆盖 |
| `无法创建工作目录` | `case_dir` 不可写（`/org` 下的权限/配额） | 确认写的是 `$R` 口径路径；`/sai_stor_tmp` 只读 |
| 全部 case 秒级 `FAILURE` + `prepared 数据不存在` | 清单 `case_dir` 与真实目录不一致（本地路径 vs `/org` 口径） | 清单里写 `/org/...` 绝对路径，别写 `~` 或本地路径 |
| `QOSMinGRES` | 4 卡作业用了 `*-1o2gpu`（每 job 上限 2 卡） | 用 `rush-gpu`/`huge-gpu`；单卡才是 1o2gpu |
| 作业 1 秒内 `CANCELLED` | 提交时用了 `sbatch --export` | 参数改走位置参数/环境变量/参数文件 |

**调参建议**：`--concurrency` 起步 2，看 `farm_summary.tsv` 的 `critical path` 与
`case-seconds`（= 估计卡时）；`--threads` 取 `DefCpuPerGPU / concurrency` 附近
（4V100 是 8 线程/GPU，并发 2 时 `--threads 4`）。

### 5.7 汇总、锚定与 TS QC

终态后按三步复核；`aggregate` 只说明有可匹配 TS/energy，`collect_ts_qc` 才检查
鞍点证据：

```bash
PY=/home/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=$R/Recnet${PYTHONPATH:+:$PYTHONPATH}

python ../tools/aggregate_network.py --root $R/recnet-runs/<campaign> --out <report_dir>
python ../tools/collect_ts_qc.py --root $R/recnet-runs/<campaign> --out <report_dir>/ts_qc.json
python ../tools/c2_gate.py \
  --campaign-root $R/recnet-runs/<campaign> \
  --aggregate-json <report_dir>/network_summary.json \
  --qc-json <report_dir>/ts_qc.json \
  --out <report_dir>/c2_gate.json \
  --coverage-threshold 0.95
python ../tools/make_network_report.py \
  --summary <report_dir>/network_summary.json \
  --anchors $R/validation-pipeline/reaction/paper_network_table.json \
  --out <report_dir>/paper_anchored_network.md
```

`ts_qc.json` 对每条 raw TS record 抽取：显著虚频数、主虚模断键/漂移投影、反应键
长度 guard、imag+ / imag− endpoint 能量，以及 pipeline 记录的非目标断键。`qc_pass`
要求单一显著虚频、断键主导、键长 guard 内、一个可解析虚频，且两个 endpoint 能量
均低于 TS。endpoint 失败者不自动删除，但应标记为“需复核/降权”，不得直接用于 C3 决策。

`c2_gate.py` 把 prepared 输入的 42 通道全集、aggregate 的 accepted rows、TS QC 和 farm
终态 manifest 合成一个机读门禁。只有 `gate_pass=true` 才进入 C3 裁决；coverage 统计
默认只计入 full QC pass 的通道。

多项实跑收口可直接用：

```bash
Recnet/recnet_adapter/tools/finalize_network.sh \
  --root $R/recnet-runs/pilot \
  --root $R/recnet-runs/campaign-c2 \
  --expected-root $R/recnet-runs/campaign-c2 \
  --report $R/recnet-runs/reports/<final-tag> \
  --anchors $R/validation-pipeline/reaction/paper_network_table.json \
  --coverage-threshold 0.95
```

`--root` 是参与聚合/QC 的数据；`--expected-root` 定义 C2 通道全集；脚本自动生成
summary、`ts_qc.json`、`c2_gate.json/.md`、论文锚定报告和 `SHA256SUMS`。

### 5.8 per-site TS seed 扫描（C–O 复跑入口）

`04_seed_scan.sbatch` 在一个 4×V100 allocation 内并发执行多个
`make_seeds_scan.py`。计划 TSV 只需三列 `arm/site/vg`（`arm=S/M`；case 为
`$RECNET_SCAN_CASE_ROOT/${RECNET_SCAN_SPECIES:-CO}-${arm}`）：

```bash
sbatch 04_seed_scan.sbatch \
  $R/recnet-runs/seed-scan-co-v1/CO_SEED_SCAN_PLAN.tsv seed-scan-co-v1
```

非 CO 物种要显式传物种，例如 CC-M 计划：

```bash
RECNET_SCAN_SPECIES=CC sbatch 04_seed_scan.sbatch \
  $R/recnet-runs/seed-scan-cc-m-v1/CC_M_SEED_SCAN_PLAN.tsv seed-scan-cc-m-v1
```

产物：

```text
$R/recnet-runs/<run_id>/seed_scan_summary.tsv
$R/recnet-runs/<run_id>/logs/<arm>-site<site>-vg<vg>.log
<case>/rxn/seeds/<rxn_key>_site_<site>[_vg<vg>].xyz
<case>/rxn/seeds/SCAN_MANIFEST_<rxn_key>_site_<site>[_vg<vg>].json
```

正式扫描前可先设 `RECNET_SCAN_DRY_RUN=1`；该模式只解析路径和扫描点，不建 DP、
不写 seed。seed 会被 `handlers/ts.py` 的 per-site override 自动读取；无 seed 的位点
行为不变。

### 5.9 campaign 终态自动收口

campaign farm 提交后，可将 C2 finalizer 挂在其任意终态之后：

```bash
sbatch --account=pku-jianghong --dependency=afterany:<campaign_job_id> \
  05_finalize_c2.sbatch
```

`05_finalize_c2.sbatch` 在 CPU-MISC 上运行，自动聚合 pilot、CO recovery 与 campaign；
产出 combined aggregate、`ts_qc.json`、`c2_gate.json/.md`、论文锚定报告、工具与报告
checksum。只有最终 `c2_gate.json` 的 `gate_pass=true` 才进入 C3 裁决。

若 gate FAIL 后要启动已准备好的 targeted round2，可把它挂在 finalizer 成功之后：

```bash
sbatch --account=pku-jianghong --dependency=afterok:<finalizer_job_id> \
  06_round2_if_needed.sbatch
```

该作业读取 finalizer 的 `c2_gate.json`：PASS 时直接退出；FAIL 时启动
`campaign-c2-round2/cases.tsv`。round2 结束后必须再次运行 finalizer 重算 gate。
