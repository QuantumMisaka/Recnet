# 师弟版 CCQN 与共享实现（MACE-Relax-Kit / mlip-agent 侧）的差异记录

> 记录日期：2026-09-16。对比对象：
> - **A** = `Recnet/ccqn/`（ResonsPomelo；仓库首个提交 `bc58973` 2026-07-16 引入该目录）
> - **B** = `MACE-Relax-Kit/algo/ccqn/`（本地副本 `~/work/sidereus/workplace/app-tools/kit/…`，
>   2026-08-31 快照；paimon 仓库是编排层，其 CCQN 实现即来自这套共享代码）
>
> 方法：逐文件 `diff` + 关键路径人工核对（证据命令见文末）。

## 1. 逐字节相同（原样沿用）

| A | B | 结论 |
|---|---|---|
| `ccqn/saddle/prfo_solver_ccqn.py` | `components/prfo_solver_ccqn.py` | **完全相同**：secular-equation 求解 RFO 子问题（dim>5）、增广 Hessian 回退、brentq 求 trust-radius 根、步长超界回缩 |
| `ccqn/utils/trust_manager.py` | `components/trust_manager_ccqn.py` | **完全相同**：rho 判据（±1/1.035、5.0）、σ=√1.15 / √0.65、边界步 vs 内部步 |

## 2. 逻辑相同、组织/实现不同

- **模式切换**（uphill ↔ prfo）：阈值一致（λ_min < −1e-6 → prfo；> +1e-2 → uphill，切到 prfo 时重置 trust radius）。
  A 将其内联为 `CCQN.select`；B 拆成 `CCQNModeSelector` 组件。
- **收敛判据**：`fmax < 阈值 且 mode == 'prfo'`，一致（A 内联进 `CCQN.converged`）。
- **StepContext**：字段一致，A 仅重排格式并加 docstring。
- **TS-BFGS Hessian 更新**：数学等价、实现不同。B 用本征基捷径 `z = |B|s`（`eigh` 后 O(N²)，不显式构造 B̃）；
  A 显式构造 `B̃ = V·diag(|w|)·Vᵀ`（多一次 O(N³) 矩阵积与 N×N 临时量）——A 是较早的写法。

## 3. 师弟新增 / 改动（相对 B）

1. **Sella 集成（B 完全没有）** —— `ccqn/saddle/sella_phase.py` + `CCQN(saddle_optimizer='sella', sella_kwargs=…)`
   - 把 `sella.Sella` 包装成与 PRFO 同接口的 phase：每步调用 `sella.step()` 得到位移 `s_k`，
     并从 Sella 的 `delta/delta0` 同步 trust radius；
   - **失败降级链**：step 异常 → 以 `delta0 = 0.5× / 0.25× / 0.1×` 重建重试；仍失败则抛
     `SellaStepFailure(suggested_radius=…)`，主循环捕获后**回退 PRFO** 并采用建议半径；
   - 工作流实际用法：`handlers/ts.py` 传 `saddle_optimizer="sella"`、`sella_kwargs={"delta0": 0.05}`，
     `run(fmax=0.05, steps=100)`；过拉伸时还有一轮 `delta0=0.02` 的保守重试。
2. **上坡方向守卫（bond-direction flip）** —— `ccqn/uphill/uphill.py::CCQNUphill.step`
   - 算出 cone-trust-region 步后，检查**第一个 reactive bond** 是否被拉长；若该步反而缩短目标键，
     则用 `−e_vec` 重算一步，取让键长增加的一侧（日志 `Uphill note: flip e-vector sign to avoid bond shrinking`）；
   - B 的 `UphillPhase.run` 无此检查（只做 e-vector 与本征向量的 overlap 诊断）。
3. **e-vector 只用 IC（简化）**：A 的上坡固定调用 `evec_ic(..., ic_mode='democratic')`；
   B 同时支持 `interp`（线性/IDPP，需要 `product_atoms`）与 `weighted` IC。
   A 的构造器仍保留 `e_vector_method/idpp_images/use_idpp/product_atoms` 形参但已不生效。
   适配动机：RMG 反应只给「变化的那根键」，不需要产物结构。
4. **每步诊断日志**：A 记 `StepInfo: |dR|=…, d(reactive)=… Å`，步日志带 `SaddleOpt=`；
   B 记 e-vector/本征向量 overlap（A 未保留）。
5. **目录扁平化**：B 的 `components/phases/contexts/gpu_components` 四层拆件在 A 合并为
   `ccqn.py + saddle/ + uphill/ + utils/`；去掉 `shared.interp` 依赖与 GPU 变体（B 另有 `ccqn_optimizer_gpu.py`）。
6. **遗留死代码**：`ccqn/uphill/utils/{environment.py,hessian.py}` 未被引用
   （`uphill/utils/__init__.py` 里 import 已注释；那份 hessian 还是更旧的 `update(B, s, y)` 签名）。

## 4. 结论与影响

- 算法骨架（uphill + PRFO + trust region + ts-BFGS + 模式切换）与共享实现一致；
  **属于师弟的实质改进是 (1) Sella 适配与失败降级、(2) 上坡键长方向守卫**。
- 两点对「表面反应 TS 搜索」这种强约束场景都有价值：
  (1) 在 PRFO 之外给了第二条路，同时用降级链保住稳健性；
  (2) 防止上坡阶段反而压缩目标键（强耦合表面体系中常见），配合 `handlers/ts.py` 的键长 QC 形成双保险。
- 相对 B 缺少：TS-BFGS 本征基加速、`interp/IDPP` 路径、overlap 诊断、GPU 组件。
- 若考虑回贡上游：建议把 (1)(2) 作为可选特性提交（`SellaPhase` 已解耦为独立文件，易移植）；
  (3) 的「只 IC」不建议回贡，恢复 `interp` 支持更通用。

## 5. 复现命令

```bash
REF=~/work/sidereus/workplace/app-tools/kit/MACE-Relax-Kit/algo/ccqn
REC=<本仓库>/ccqn
diff -q $REF/components/prfo_solver_ccqn.py $REC/saddle/prfo_solver_ccqn.py   # 相同
diff -q $REF/components/trust_manager_ccqn.py  $REC/utils/trust_manager.py    # 相同
diff -u $REF/components/hessian_manager.py     $REC/utils/hessian.py          # TS-BFGS 实现差异
```
