# Recnet × deepmd-kit 3.2+ / DPA4 兼容性梳理

> 整理日期：2026-09-16。对象：`Recnet`（本 fork 的 `feat/ft2dp-dpa4-adapter` 分支）对 deepmd-kit ≥3.2 的
> DPA4（pt 后端）模型消费能力。证据标注了来源（源码阅读 / 本地实测 / SAI 作业）。

## 0. 结论

- Recnet 对 deepmd 的**接口面极窄**：`DP` 构造 + `get_potential_energy`/forces +（包装器）改 `results`/`parameters`。
  这些在 deepmd-kit 3.2 下全部可用，**不需要改 Recnet 的算法代码**。
- 真正的兼容风险不在 API，而在三件事：**①多头模型的默认头**、**②环境/模型构建版本对齐**、**③少数未用到的
  深接口（get_hessian / charge_spin）**。适配器已处理 ①，②③ 见 §3。
- 生产路径 = SAI `dpeva-dpa4`（deepmd `3.2.0b1.dev67` + torch `2.11.0+cu126` + CUDA 12.6.3），
  已用真实作业端到端验证（job `1342005` = PASS，能量与本地逐位一致）。

## 1. 接口面清单（Recnet 侧实际用到的 deepmd/ASE API）

| Recnet 用法 | 出现处 | deepmd-kit 3.2 状态 | 证据 |
|---|---|---|---|
| `from deepmd.calculator import DP` | 5 个模块（handlers×4 + utils/constraints） | ✓ | 源码 |
| `DP(model=MODEL)` | **30 处** | ✓ | 全流程；适配器在导入前注入 `head` |
| `DP(model=..., head="ft2dp")` | 适配器注入 | ✓（3.2 `DP.__init__` 有 `head` 形参） | 本地 check + SAI 作业 |
| `class HarmonicallyForcedDP(DP)`：调用 `DP.calculate()` 后对
  `self.results["energy"/"free_energy"/"forces"] +=` | 5 处 | ✓（3.2 的 `calculate()` 写这 3 个键，另写 virial/stress） | 读源码 + 包装器路径实测（差值 0.004 meV） |
| `atoms.calc.get_potential_energy()`（优化器/振动内部再调 `get_forces`） | 20 处 | ✓ | 冒烟 + 作业 |
| `calc.parameters.<自定义键>`（约束势透传） | 仅包装器 | ✓（ASE 基类参数透传） | 包装器实测 |
| **未用到**：`get_hessian` / `charge_spin` / `type_dict` / `neighbor_list` / `stress` | — | — | grep 全仓确认 |

## 2. 版本矩阵（已实测）

| 组合 | 结论 | 证据 |
|---|---|---|
| deepmd `3.2.0b1.dev67` + DPA4 多头 ckpt（`6152d7b2…`） | **PASS**（能量/力、head 注入、包装器、时延全绿） | 本地三场景 check；SAI 作业 `1342005`（V100，16 原子 20.2 ms/次） |
| deepmd `3.2.0` GA（SAI `dpeva-dpa4-320`）+ 同一 ckpt | ✅ **与 dev67 数值逐位一致**：CO −611.9102 / Fe₈ −51526.5419 / Default −16.313 | SAI 实测（2026-09-16，见 §6） |
| 旧 DPA2.3.1 `.pth`（师弟现后端） | 未回归（本工作流不需要）；其集群仍需自建 3.2+ 环境才能换 DPA4 | — |

## 3. 关键兼容点（按风险排序）

1. **默认头陷阱（最高危，已在适配器修复）**：多头 ckpt 上 `DP(model=…)` 不带 `head` 会静默使用 `Default` 头；
   FT2DP 的 `Default`（MatPES replay）与 `ft2dp` 头**能量基线不同**（实测 CO：−16.31 vs −611.91 eV；
   Fe 体系差 ~5×10⁴ eV）。适配器注入 `head`，且未知头名**硬失败**（deepmd 报 `Available heads are: ['Default','ft2dp']`）。
2. **`use_compile` 与首次调用延迟**：ckpt 配置 `use_compile: true`；环境未设 `DP_COMPILE_INFER=0` 时，
   推理路径会进 torch.compile —— 登录节点实测出现分钟级卡顿（进程活着但不出结果）。
   SAI 生产 env 脚本已设 `DP_COMPILE_INFER=0`，实测稳态 ~20 ms/次（V100，16 原子）。
3. **checkpoint 键版本偏移**：2026-07 训练的 ckpt 在较新 dev 构建上加载会报 missing keys
   （`so2_conv.coeff_index_m` / `degree_index_m` / `rotate_inv_rescale_full` / `_empty_tensor`），**推理数值正常**
   （与训练标签一致到 meV/atom）；但同一次加载在 `dp --pt freeze` 的**严格 state_dict 加载**下会失败
   → 单头导出必须用与训练一致的构建版本复验（P7 新模型上再试）。
4. **`get_hessian` 在 DP 中不存在**：`ccqn` 的 `HessianUpdater(model='calc')` 会调 `calc.get_hessian()`，
   deepmd 3.2 的 `DP` 没有该方法。Recnet 默认 `hessian_model='ts-bfgs'`（单位阵×70 起步）**不受影响**；
   若有人显式切到 `'calc'` 会直接崩 —— 属于「不要打开的开关」。
5. **spin / charge_spin**：3.2 的 `DP.calculate()` 读取 `atoms.info["charge_spin"]`；Recnet 从不设置。
   当前 FT2DP 非 spin（P7 输入无 spin 键）→ 无影响；若启用 native-spin 消融（Q7），需要给 Recnet 补 `charge_spin` 传参。
6. **能量标度**：`ft2dp` 头是 ABACUS 绝对能量标度（~−1800 eV/atom 量级）；Recnet 只用差值（势垒/吸附能），
   自洽；但**跨模型/跨头绝对能量不可比**，对照实验必须锁定同一头。
7. **元素覆盖**：DPA4 基座 type_map=118 元素、observed=89；FT2DP 训练域仅 C/Fe/H/O。
   域外元素「能跑」但精度无担保（`preflight_case.py` 会 WARN）。
8. **邻居表**：DPA4/SEZM 用内建 neighbor list（vesin）。本地 WSL 需要 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`
   （libcuda.so）；SAI env 脚本已把 CUDA lib 加进 `LD_LIBRARY_PATH` → 集群无需特殊处理。
9. **GPU 架构**：V100（sm_70，env 脚本 `TORCH_CUDA_ARCH_LIST=7.0`）✓；本地 2070S（sm_75）✓。
10. **并行/精度环境变量**：`DP_INTERFACE_PREC=high`、`OMP_NUM_THREADS` 等由 `$R/dpeva-git/scripts/env/dpeva-dpa4.env`
    统一设置；未见阻塞项。

## 4. 部署硬要求（清单）

- deepmd-kit **≥ 3.2.0**（pt 后端）+ torch（实测 `2.11.0+cu126`）；env 建议直接用
  `$R/dpeva-git/scripts/env/dpeva-dpa4.env`（含 CUDA module、`DP_VARIANT=cuda`、`DP_COMPILE_INFER=0`）。
- 模型：与训练构建一致的 checkpoint（本工作流用 `model.ckpt-*.pt`；SHA 已登记于
  `recnet_adapter/evidence/README.md` 与 `sai/README.md`）。
- 资源：**单卡**即可（Recnet 是单进程 ASE 流程，无多卡路径）；≤255 原子为 v1 QC 口径，管线无硬上限。

## 5. 未验证 / 待办

- ~~GA `3.2.0` 与 dev67 的逐数值一致性~~ → **已完成**（2026-09-16，见 §6）。
- `dp --pt freeze --head ft2dp` 在 **P7 新 ckpt + 同版本环境**下是否可用（单头导出备选路径）。
- 若要消费 **DPA4C / OpenLAM 压缩版**（P2-B 的 pt-expt / eval-desc 限制），需单独评估（当前路径不涉及）。

## 6. 记录（追加）

- **2026-09-16 跨版本一致性（GA vs dev67）**：SAI 上用生产 env 脚本
  （`DPEVA_DPA4_ENV_NAME=dpeva-dpa4-320 source $R/dpeva-git/scripts/env/dpeva-dpa4.env`）对同一 ckpt
  `model.ckpt-100000.pt`（SHA `6152d7b2…`）做单点，与 dev67 结果逐位一致：

  | 体系 | dev67 | GA 3.2.0 |
  |---|---|---|
  | CO（`head=ft2dp`） | −611.9102 eV | **−611.9102 eV** |
  | Fe bcc 2×2×2（`head=ft2dp`） | −51526.5419 eV | **−51526.5419 eV** |
  | CO（默认 `Default` 头） | −16.3130 eV | **−16.313 eV** |

  结论：**deepmd-kit 3.2 系列（GA 3.2.0 与 3.2.0b1.dev67）对同一 DPA4 ckpt 的消费数值一致**，
  环境选择不影响管线结果；注意 GA 环境下登录节点 CPU 推理较慢（3 次单点约 4–5 分钟），
  生产上按 env 脚本 + GPU 作业执行即可。
