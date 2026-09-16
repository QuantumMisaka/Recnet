# tools/ — case 准备与预检

| 脚本 | 用途 |
|---|---|
| `build_slab.py` | 从 bulk 结构切任意 Miller 指数表面板（pymatgen），输出 Recnet 就绪的 slab CIF + 几何报告 + `--bottom-freeze-threshold` 建议 |
| `preflight_case.py` | 提交前预检 case 目录（slab + prepared_data）：字段/索引/元素/几何/（可选）DP 单点 |
| `slab_utils.py` | 两者复用的几何助手（分层、最近距离、元素覆盖、法向对齐） |

## 典型流程

```bash
# 0) 依赖：pymatgen 仅在 build_slab 需要（SAI: `ase`/`atst` env 已含；本机: pip install pymatgen）

# 1) 列出 (510) 的所有终止，挑一个 Fe 终止面
python build_slab.py Fe5C2_bulk.vasp --miller 5 1 0 --list

# 2) 生成 slab（3x2 扩胞、~4 层、12 Å 真空）
python build_slab.py Fe5C2_bulk.vasp --miller 5 1 0 --termination 0 \
    --size 3 2 --layers 8 --vacuum 12 --out $R/recnet-runs/fe5c2-510-ft/Fe5C2_510.cif

# 3) 把 Fe2C(110) case 的 prepared_data/ 拷进来（反应网络与表面解耦，可复用）
cp -r $R/recnet-runs/fe2c110-case/prepared_data $R/recnet-runs/fe5c2-510-ft/

# 4) 预检
python preflight_case.py $R/recnet-runs/fe5c2-510-ft --dp-check

# 5) 通过后提交
RECNET_CASE=$R/recnet-runs/fe5c2-510-ft \
RECNET_SLAB=$R/recnet-runs/fe5c2-510-ft/Fe5C2_510.cif \
RECNET_EXTRA_ARGS="--use-c-vacancy-io --bottom-freeze-threshold <报告建议值>" \
  sbatch ../sai/01_run_pipeline.sbatch
```

## 约定与注意

- **z 轴朝上 + Fe 表层**：`handlers/context.py` 用「最高 Fe 层 z_max − 0.1」定义表层，并在 Fe 原子中
  找 Voronoi 位点。C 终止面会拿不到位点——若确需 C 终止，先联系维护者加 `RECNET_TOP_ELEMENT` 开关。
- `prepared` 数据里 `broken_bond` 是**对 reactant_species[0] 模板**的索引（`handlers/ts.py` 直接如此使用）；
  预检会检查越界并在报错信息里提示 prepare 阶段的常见错因。
- 冻结阈值：默认冻结底部 2 层；预检/建板脚本都会给出建议值，冷启动别用管线内置默认。
- `--dp-check` 建议在 GPU 节点跑（登录节点慢）；它只做单点，不做弛豫。
