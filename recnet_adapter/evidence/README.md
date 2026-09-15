# recnet_adapter 验证证据（2026-09-15）

- 运行环境：deepmd-kit `3.2.0b1.dev67+g73de44b1f` | torch `2.11.0+cu126` | CUDA `True`（WSL2, RTX 2070 SUPER）
- 命令均在 `Recnet/` 目录下执行；日志为原始输出（含 deepmd 加载警告，未删改）
- 使用的两个检查点（同基座 `DPA4-Air-ZBL-MatPES-v20260629`）：
  - 多头：`model/ft2dp-dpa4/multitask-forgetting-20260711/ft2dp-dpa4-air-multitask-replay-lr350e-3-regular-100k.pt`
    SHA256 `6152d7b2f540bfe07a33147714ee5b7afbfd50cec08971474a29c414c68ce8e0`
  - 单头：`model/ft2dp-dpa4/ft2dp-dpa4-air-regular-50k.pt`
    SHA256 `d1cbb174bd957ce8b0d027e24c3ccdb07aa1c0bb267378b3e5d004232dc288e9`

| 证据文件 | 命令 | 结果 |
|---|---|---|
| `check_multitask_ft2dp_20260915.log` | `RECNET_DP_MODEL=../model/.../ft2dp-dpa4-air-multitask-replay-lr350e-3-regular-100k.pt python -m recnet_adapter check --repeats 3` | **PASS**，退出码 0；管线式 vs 显式 head 差 ≤0.01 meV；`Default` 头基线差 ~5×10⁴ eV（head 注入生效证据） |
| `check_singlehead_20260915.log` | `RECNET_DP_HEAD=none RECNET_DP_MODEL=../model/ft2dp-dpa4/ft2dp-dpa4-air-regular-50k.pt python -m recnet_adapter check --repeats 2` | **PASS**，退出码 0（单头路径，不注入 head） |
| `check_badhead_failsafe_20260915.log` | `... python -m recnet_adapter check --head nope` | **FAIL**，退出码 1；报错列出可用头 `['Default', 'ft2dp']`（无静默回退） |
| `wrapper_entry_help_20260915.log` | `python recnet_adapter/run_dp_ts_ft2dp.py --help` | 映射先安装（head=ft2dp + MODEL 重绑定 5 处），随后进入 `run_dp_ts.py`，退出码 0 |
| `sai_gpu_check_20260916.log` | SAI 作业 `1342005`（4V100 / 节点 4v100n34 / `rush-1o2gpu`）：`python -m recnet_adapter check --repeats 5` | **PASS**（1:47 完成，cuda True）：能量与本地逐位一致（≤0.006 meV）；`Default` 头差 ~5×10⁴ eV；16 原子 Fe(110) **20.2 ms**/E+F（V100，约本地 RTX 2070S 的 5×） |

复现：重跑上述任一命令即可；`check` 退出码 `0`=PASS / `1`=FAIL / `2`=配置错误 / `3`=运行时错误。
