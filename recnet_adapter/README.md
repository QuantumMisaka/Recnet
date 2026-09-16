# recnet_adapter — 把 FT2DP（DPA4）模型映射到 Recnet 管线

## 这是什么

Recnet 把后端模型硬编码在 `handlers/context.py`（`MODEL = "<...>.pth"`，旧 DPA2.3.1），
并在 5 个模块里用 `DP(model=MODEL)` 构造 ASE 计算器。本适配器**不改管线源码**，完成两件事：

1. **head 注入**：在 `handlers` 首次导入前，把 `deepmd.calculator.DP` 替换为默认携带
   `head=<配置值>` 的子类；此后管线所有 `DP(model=MODEL)`、以及自带的
   `HarmonicallyForcedDP(...)` 包装器都会自动带上正确的头。
2. **MODEL 注入**：把 `handlers.context.MODEL` 及各 handler 模块导入时复制走的同一常量，
   改写为配置的 FT2DP 检查点。

## 快速开始

```bash
# 0) 环境：需 deepmd-kit >= 3.2.0（pt 后端 + torch）
conda activate <环境>

# 1) 指定模型（也可写进 model_config.json）
export RECNET_DP_MODEL=/path/to/ft2dp-dpa4.pt   # 多头检查点（Default + ft2dp）
export RECNET_DP_HEAD=ft2dp                     # 单头/冻结模型请设 none

# 2) 查看映射结果（不加载模型）
python -m recnet_adapter show

# 3) 端到端校验（换模型后建议都跑一次）
python -m recnet_adapter check

# 4) 跑管线：参数与 run_dp_ts.py 完全一致，原样透传
python -m recnet_adapter run --path ./ --prepared ./prepared_data/prepared_rmg_data.yaml --slab ./surf.cif --top-x 1
# 或 sbatch 场景（等价）：
python recnet_adapter/run_dp_ts_ft2dp.py --path ./ --prepared ... --slab ...
```

配置优先级：**命令行参数 > 环境变量 `RECNET_DP_MODEL` / `RECNET_DP_HEAD` > `model_config.json` >
管线内置默认（旧 DPA2.3.1，即适配器不生效）**。

## 为什么必须显式 head

2026-09-15 实测：同一检查点、同一体系，`Default` 头与 `ft2dp` 头**能量基线不同**
（例：CO 气相 `ft2dp` = −611.91 eV，`Default` = −16.31 eV；Fe 体系差 ~5×10⁴ eV）。
若不带 `head`，deepmd 会静默选 `Default` —— 对 FT2DP 用途是**错误结果**。

## `check` 输出怎么读

| 列 | 含义 |
|---|---|
| `pipeline-E` | 走管线调用式（补丁后的 DP + 改写后的 MODEL） |
| `ref(head)-E` | 显式 `DP(model=..., head='ft2dp')` 参考 |
| `\|diff\|/meV` | 应 ≈ 0（float32 在 ~5×10⁴ eV 量级上有 ≤0.01 meV 舍入，属正常） |
| `default-head-E` | 不带 head 的能量；多头模型下应与前两列差异巨大 = head 注入生效的证据 |

另有 `HarmonicallyForcedDP`（管线自带约束计算器）与 E+F 时延两项。退出码：
`0`=PASS，`1`=FAIL，`2`=配置错误，`3`=运行时错误（`RECNET_ADAPTER_DEBUG=1` 看完整堆栈）。

## 环境要求

- deepmd-kit **>= 3.2.0**（pt 后端）+ torch（本地验证：`3.2.0b1.dev67` + `torch 2.11.0+cu126`）
- 管线依赖：`ase sella scipy rdkit molecule networkx yaml`
- WSL 本机运行注意：`export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`（vesin 邻居表要 `libcuda.so`）

## 已知事项

- **上游导入 bug（已在本克隆修复，建议回流上游）**：`handlers/{workflow,context,adsorption,energy,ts}.py`
  残留重构后的 `from ..utils import ...` 相对导入（`utils/` 里已无 adsorption/ts/irc/energy），
  HEAD 无法 `import handlers`。已改为同包/绝对导入（共 12 行，仅导入语句）——使用本适配器的前置修复。
- **单头导出**：`dp --pt freeze --head ft2dp` 在 2026-07 旧检查点上因版本键差失败（推理可加载、
  freeze 严格加载失败），故采用 head 注入；P7 新模型可在同版本环境复验 freeze 作为「零改动」替代。
- **能量标度**：不同头/不同模型的绝对能量不可比（本区 FT2DP 标签为 ABACUS 绝对能量标度）。
  管线只使用差值（势垒、吸附能），同一 head 内自洽。

## 关联文档

- `../../docs/tasks/reaction-network-dpa4.md`：ft2dp-dpeva 工作区中本工作流的事实源（DoD/阶段/Ruling）

## 在 SAI 集群运行

见 `sai/README.md`（同步、作业脚本 `00_backend_check.sbatch` / `01_run_pipeline.sbatch`、
输入目录规范、集群模型清单与 SHA）。要点：单卡 `4V100` + `rush-1o2gpu`/`flood-1o2gpu`，
只给 `--nodes/--ntasks/--gpus-per-node` 三参数，路径用 `/org/...`（`$R`）口径。

## 相关材料

- `tools/`：case 准备工具 —— `build_slab.py`（任意 Miller 切面 + z 轴对齐 + 冻结阈值建议）、
  `preflight_case.py`（提交前预检，支持 `--dp-check`）。用法见 `tools/README.md`。
- `NOTES-ccqn-vs-upstream.md`：师弟版 `ccqn/` 与共享实现（MACE-Relax-Kit / mlip-agent 侧）的逐文件差异记录。
