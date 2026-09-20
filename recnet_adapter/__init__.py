"""
recnet_adapter — 把 FT2DP（DPA4）模型映射到 Recnet 反应网络管线。

背景
----
Recnet 管线把后端模型硬编码在 ``handlers/context.py``（``MODEL = "<...>.pth"``，
DPA2.3.1 多头检查点），并在 5 个模块里用 ``DP(model=MODEL)`` 构造 ASE 计算器。
对多头 DPA4 检查点（头为 ``Default`` + ``ft2dp``）而言，不带 ``head`` 参数会静默
选用 ``Default`` 头 —— 它的能量基线与 ``ft2dp`` 头完全不同，会得到错误结果。

本适配器在不修改管线源码的前提下完成映射：

1. **head 注入**：在 ``handlers`` 被导入之前，把 ``deepmd.calculator.DP`` 替换为
   默认携带 ``head=<配置值>`` 的子类；此后所有 ``from deepmd.calculator import DP``
   的模块（handlers/*、utils/constraints）都会继承，包括管线自带的
   ``HarmonicallyForcedDP`` 包装器。
2. **MODEL 注入**：把 ``handlers.context.MODEL`` 及被各 handler 模块在导入时复制走的
   同一常量，改为指向配置的检查点。

配置优先级：显式参数 > 环境变量（``RECNET_DP_MODEL`` / ``RECNET_DP_HEAD``）>
``model_config.json`` >（不改动，保持管线内置默认模型）。

用法::

    python -m recnet_adapter show          # 查看当前映射解析结果
    python -m recnet_adapter check         # 端到端校验（推荐先跑）
    python -m recnet_adapter run --path . --prepared ... --slab ...
    # 或（等价，便于放进 sbatch）：
    python recnet_adapter/run_dp_ts_ft2dp.py --path . --prepared ... --slab ...
"""
from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__version__ = "0.1.0"

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
CONFIG_PATH = ADAPTER_DIR / "model_config.json"

ENV_MODEL = "RECNET_DP_MODEL"
ENV_HEAD = "RECNET_DP_HEAD"

#: 管线自有的模块前缀（inject/rebind 只作用于这些命名空间）
_PIPELINE_PREFIXES = ("handlers", "utils", "ccqn", "recnet_adapter", "run_dp_ts")

#: 注入前的原始 DP 类（首次 install 时记录）
_ORIGINAL_DP: Optional[type] = None
#: 当前已注入的 head（None = 未注入）
_INJECTED_HEAD: Optional[str] = None

__all__ = [
    "AdapterError",
    "Resolution",
    "resolve",
    "install",
    "bind_model",
    "check_backend",
    "describe",
    "run_pipeline",
    "REPO_ROOT",
    "ADAPTER_DIR",
    "ENV_MODEL",
    "ENV_HEAD",
]


class AdapterError(RuntimeError):
    """适配器配置/映射错误。"""


@dataclass
class Resolution:
    """模型映射解析结果。"""

    model: Optional[Path]
    head: Optional[str]
    source: dict

    def summary(self) -> str:
        m = str(self.model) if self.model is not None else "<pipeline built-in default>"
        h = repr(self.head) if self.head else "None (single-head / default)"
        return f"model={m}\nhead={h}\nsource={self.source}"


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:  # 配置坏掉不阻塞，只警告
        print(f"[recnet_adapter] WARNING: 无法解析 {CONFIG_PATH}: {exc}", file=sys.stderr)
        return {}


def _clean_head(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null", "off", "false"):
        return None
    return text


def _resolve_model_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null"):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def resolve(model: Any = None, head: Any = None) -> Resolution:
    """按 参数 > 环境变量 > model_config.json 顺序解析模型路径与 head。"""
    cfg = _load_config()
    source = {"model": "unset", "head": "unset"}

    raw_model = None
    if model is not None:
        raw_model, source["model"] = model, "argument"
    elif os.environ.get(ENV_MODEL):
        raw_model, source["model"] = os.environ[ENV_MODEL], f"env:{ENV_MODEL}"
    elif cfg.get("model"):
        raw_model, source["model"] = cfg["model"], "model_config.json"

    raw_head = None
    if head is not None:
        raw_head, source["head"] = head, "argument"
    elif os.environ.get(ENV_HEAD):
        raw_head, source["head"] = os.environ[ENV_HEAD], f"env:{ENV_HEAD}"
    elif cfg.get("head") is not None:
        raw_head, source["head"] = cfg.get("head"), "model_config.json"

    return Resolution(
        model=_resolve_model_path(raw_model),
        head=_clean_head(raw_head),
        source=source,
    )


# ---------------------------------------------------------------------------
# 注入实现
# ---------------------------------------------------------------------------

def _make_headed_class(base: type, head: str) -> type:
    """构造 ``head`` 默认注入的子类（仅在调用方未显式传 head 时生效）。"""

    class _HeadedDP(base):  # type: ignore[misc, valid-type]
        _recnet_injected_head = head

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("head", head)
            super().__init__(*args, **kwargs)

    _HeadedDP.__name__ = "DP"
    _HeadedDP.__qualname__ = "DP"
    _HeadedDP.__module__ = base.__module__
    _HeadedDP.__doc__ = (base.__doc__ or "") + (
        f"\n\n[recnet_adapter] 已注入默认 head={head!r}（可用显式 head=... 覆盖）"
    )
    return _HeadedDP


def _rebind_imported(head_class: type, head: str) -> list:
    """把已导入的管线模块中绑定到旧 DP 的名字改绑到新类。"""
    rebinds = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(_PIPELINE_PREFIXES):
            continue
        if getattr(module, "DP", None) is _ORIGINAL_DP:
            setattr(module, "DP", head_class)
            rebinds.append(f"{name}.DP")
        harmonic = getattr(module, "HarmonicallyForcedDP", None)
        if (
            isinstance(harmonic, type)
            and issubclass(harmonic, _ORIGINAL_DP)
            and getattr(harmonic, "_recnet_injected_head", None) != head
        ):
            setattr(module, "HarmonicallyForcedDP", _make_headed_class(harmonic, head))
            rebinds.append(f"{name}.HarmonicallyForcedDP")
    return rebinds


def bind_model(model: Path) -> list:
    """把管线各模块中的 ``MODEL`` 常量重绑到给定检查点（会先导入 handlers 包）。"""
    _ensure_repo_on_path()
    try:
        importlib.import_module("handlers.context")
    except Exception as exc:  # noqa: BLE001 - 需要把缺依赖信息透传给用户
        raise AdapterError(
            "无法导入 Recnet 管线包 handlers（请确认依赖 ase/deepmd/sella/ccqn 已安装，"
            f"且从 Recnet 仓库根运行）：{exc.__class__.__name__}: {exc}"
        ) from exc

    value = str(model)
    bound = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(_PIPELINE_PREFIXES):
            continue
        if hasattr(module, "MODEL"):
            setattr(module, "MODEL", value)
            bound.append(f"{name}.MODEL")
    return sorted(bound)


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def install(
    model: Any = None,
    head: Any = None,
    *,
    require_model: bool = True,
    verbose: bool = True,
) -> dict:
    """把 FT2DP 模型映射进当前进程的 Recnet 管线。

    必须在 ``handlers`` 首次导入之前调用（``run_pipeline`` / 包装脚本已保证）。
    返回映射报告；重复调用是幂等的。
    """
    global _ORIGINAL_DP, _INJECTED_HEAD

    res = resolve(model=model, head=head)
    if res.model is not None and not res.model.exists():
        raise AdapterError(
            f"FT2DP 检查点不存在: {res.model}\n"
            f"（来源 {res.source['model']}；可用 {ENV_MODEL} 环境变量或 --model 覆盖，"
            "单头/冻结模型请在 model_config.json 里把 head 设为 null）"
        )
    if res.model is None and require_model:
        raise AdapterError(
            "未配置 FT2DP 模型（管线会继续用它内置的旧 DPA2.3.1 默认模型）。\n"
            f"请设置环境变量 {ENV_MODEL}=<检查点路径>，或编辑 {CONFIG_PATH}。"
        )

    _ensure_repo_on_path()
    report: dict = {"resolution": res, "patched": [], "rebound": [], "warnings": []}

    try:
        import deepmd.calculator as dm_calculator
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(
            "无法导入 deepmd（DPA4 需要 deepmd-kit >= 3.2.0，pt 后端 + torch）："
            f"{exc.__class__.__name__}: {exc}"
        ) from exc

    if _ORIGINAL_DP is None:
        _ORIGINAL_DP = dm_calculator.DP

    if res.head:
        if _INJECTED_HEAD != res.head:
            head_class = _make_headed_class(_ORIGINAL_DP, res.head)
            dm_calculator.DP = head_class
            _INJECTED_HEAD = res.head
            report["patched"].append(f"deepmd.calculator.DP -> head={res.head!r}")
            report["rebound"] = _rebind_imported(head_class, res.head)
        else:
            report["patched"].append(f"deepmd.calculator.DP（已注入 head={res.head!r}）")
    elif _INJECTED_HEAD is not None:
        dm_calculator.DP = _ORIGINAL_DP
        _INJECTED_HEAD = None
        report["patched"].append("deepmd.calculator.DP（恢复原始类，无 head 注入）")

    if res.model is not None:
        bound = bind_model(res.model)
        report["patched"].append(f"MODEL -> {res.model}")
        report["rebound"].extend(bound)
    else:
        report["warnings"].append("未配置模型，管线保持内置默认 MODEL")

    if verbose:
        for line in report["patched"]:
            print(f"[recnet_adapter] {line}")
        if report["rebound"]:
            print(f"[recnet_adapter] 重绑定: {', '.join(report['rebound'])}")
        for warning in report["warnings"]:
            print(f"[recnet_adapter] WARNING: {warning}", file=sys.stderr)
    return report


# ---------------------------------------------------------------------------
# 运行时入口
# ---------------------------------------------------------------------------

def run_pipeline(extra_args: list, model: Any = None, head: Any = None) -> None:
    """安装映射后，以相同 CLI 运行管线的 ``run_dp_ts.py``。"""
    _ensure_repo_on_path()
    install(model=model, head=head)
    script = REPO_ROOT / "run_dp_ts.py"
    argv = [arg for arg in (extra_args or []) if arg != "--"]
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


# ---------------------------------------------------------------------------
# 环境与校验
# ---------------------------------------------------------------------------

def _env_info() -> dict:
    info: dict = {}
    try:
        import deepmd  # noqa: F401

        info["deepmd-kit"] = getattr(deepmd, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        info["deepmd-kit"] = f"MISSING ({exc.__class__.__name__})"
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"MISSING ({exc.__class__.__name__})"
        info["cuda"] = False
    return info


def describe(stream=None) -> None:
    """打印当前解析结果（不导入管线、不加载模型）。"""
    out = stream or sys.stdout
    res = resolve()
    env = _env_info()

    print("== recnet_adapter show ==", file=out)
    print(f"adapter    : {ADAPTER_DIR}", file=out)
    print(f"env        : deepmd-kit {env['deepmd-kit']} | torch {env['torch']} | cuda {env['cuda']}", file=out)
    print(f"config     : {CONFIG_PATH}", file=out)

    if res.model is None:
        print(f"model      : <pipeline built-in default>   (source: {res.source['model']})", file=out)
    else:
        exists = "OK" if res.model.exists() else "MISSING"
        size = f"{res.model.stat().st_size / 1e6:.1f} MB" if res.model.exists() else "-"
        print(f"model      : {res.model}  [{exists}, {size}]   (source: {res.source['model']})", file=out)
    print(f"head       : {res.head!r}   (source: {res.source['head']})", file=out)

    _ensure_repo_on_path()
    try:
        ctx = importlib.import_module("handlers.context")
        print(f"pipeline   : handlers.context.MODEL = {ctx.MODEL}", file=out)
        print(f"             （未 install 时即管线内置默认值）", file=out)
    except Exception as exc:  # noqa: BLE001
        print(f"pipeline   : 无法导入 handlers（{exc.__class__.__name__}: {exc}）", file=out)


def _test_structures() -> dict:
    """构造小的 C/H/O/Fe 测试体系（含周期性表面）。"""
    from ase import Atoms
    from ase.build import bcc110, bulk

    structures = {}

    co = Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.13)])
    co.set_cell([20.0, 20.0, 20.0])
    co.set_pbc([False, False, False])
    co.center()
    structures["CO(gas)"] = co

    h2o = Atoms("OH2", positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.24, 0.927, 0.0)])
    h2o.set_cell([20.0, 20.0, 20.0])
    h2o.set_pbc([False, False, False])
    h2o.center()
    structures["H2O(gas)"] = h2o

    fe_bulk = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    structures["Fe-bcc(2x2x2)"] = fe_bulk

    slab = bcc110("Fe", size=(2, 2, 4), vacuum=8.0)
    structures["Fe(110)-slab"] = slab
    return structures


def _check_sella_runtime() -> tuple[bool, str]:
    """Sella 初始化检查；用于提前暴露 user-site/JAX/NumPy 版本冲突。"""
    try:
        import importlib.metadata as metadata

        import sella
        from ase.build import molecule

        atoms = molecule("H2O")
        atoms.center(vacuum=5.0)
        opt = sella.Sella(atoms, order=1, delta0=0.1)
        version = getattr(sella, "__version__", None)
        if version is None:
            try:
                version = metadata.version("Sella")
            except Exception:  # noqa: BLE001
                version = "?"
        detail = f"Sella {version} | {sella.__file__}"
        del opt
        return True, detail
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def check_backend(model: Any = None, head: Any = None, *, repeats: int = 3) -> bool:
    """端到端校验映射：管线调用式 vs 显式 head 参考、包装器、时延。

    返回 True 表示全部通过。
    """
    import numpy as np

    res = resolve(model=model, head=head)
    assert _ORIGINAL_DP is not None or True  # 下面 install 会填充

    print("== recnet_adapter check ==")
    env = _env_info()
    print(f"env        : deepmd-kit {env['deepmd-kit']} | torch {env['torch']} | cuda {env['cuda']}")
    report = install(model=model, head=head)
    res = report["resolution"]

    dm_calculator = sys.modules["deepmd.calculator"]
    patched_dp = dm_calculator.DP
    ctx = importlib.import_module("handlers.context")
    constraints = importlib.import_module("utils.constraints")
    model_used = ctx.MODEL
    print(f"model      : {model_used}")
    print(f"head       : {res.head!r}  (patched: {patched_dp is not _ORIGINAL_DP})")
    print("")

    def evaluate(atoms, calc):
        frame = atoms.copy()
        frame.calc = calc
        return float(frame.get_potential_energy())

    def tolerance(e_ref: float) -> float:
        """float32 推理在大总量能量上有 ~1e-9 相对舍入；给 1e-5 eV 下限即可。"""
        return max(1e-5, 1e-9 * abs(e_ref))

    structures = _test_structures()
    rows = []
    all_ok = True

    header = f"{'structure':<16}{'pipeline-E':>15}{'ref(head)-E':>15}{'|diff|/meV':>12}{'default-head-E':>18}"
    print(header)
    print("-" * len(header))

    head_effect_seen = False
    had_error = False
    for label, atoms in structures.items():
        try:
            e_pipe = evaluate(atoms, patched_dp(model=model_used))
            if res.head:
                e_ref = evaluate(atoms, _ORIGINAL_DP(model=model_used, head=res.head))
            else:
                e_ref = evaluate(atoms, _ORIGINAL_DP(model=model_used))
            try:
                e_default = evaluate(atoms, _ORIGINAL_DP(model=model_used))
            except Exception:  # noqa: BLE001 - 单头模型无歧义，忽略
                e_default = float("nan")
        except Exception as exc:  # noqa: BLE001 - 报清楚而不是抛栈
            all_ok = False
            had_error = True
            print(f"{label:<16}  ERROR: {exc.__class__.__name__}: {exc}")
            continue

        diff = abs(e_pipe - e_ref)
        if np.isfinite(e_default) and abs(e_default - e_ref) > tolerance(e_ref):
            head_effect_seen = True

        ok = diff <= tolerance(e_ref)
        all_ok &= ok
        rows.append((label, e_pipe, e_ref, diff, e_default, ok))
        flag = "OK" if ok else "FAIL"
        print(f"{label:<16}{e_pipe:>15.4f}{e_ref:>15.4f}{diff * 1e3:>12.3f}{e_default:>18.4f}  {flag}")

    print("")
    if res.head and not head_effect_seen and not had_error:
        print("[warn] 未观察到 head 差异：模型可能是单头（正常），也可能 head 注入失效（请人工确认）")

    # --- Sella 运行时（Recnet TS/CCQN 必经路径；backend check 不能只测 E/F）---
    sella_ok, sella_detail = _check_sella_runtime()
    print(f"Sella init          : {sella_detail}  {'OK' if sella_ok else 'FAIL'}")
    all_ok &= sella_ok

    # --- Recnet 自带包装器 HarmonicallyForcedDP（表面积点位点约束路径）---
    slab = structures["Fe(110)-slab"]
    site_pos = slab.positions[0].copy()
    wrapper_ok = False
    try:
        wrapper_calc = constraints.HarmonicallyForcedDP(
            model=model_used,
            atom_bond_potentials=[],
            center_bond_potentials=[],
            site_bond_potentials=[{"site_pos": site_pos, "ind": 5, "k": 0.5, "deq": 0.0}],
        )
        e_wrap = evaluate(slab, wrapper_calc)
        ref_kwargs = {"head": res.head} if res.head else {}
        ref_calc = constraints.HarmonicallyForcedDP(
            model=model_used,
            atom_bond_potentials=[],
            center_bond_potentials=[],
            site_bond_potentials=[{"site_pos": site_pos, "ind": 5, "k": 0.5, "deq": 0.0}],
            **ref_kwargs,
        )
        e_wrap_ref = evaluate(slab, ref_calc)
        wrap_diff = abs(e_wrap - e_wrap_ref)
        wrapper_ok = wrap_diff <= tolerance(e_wrap_ref)
        print(f"HarmonicallyForcedDP : E={e_wrap:.4f}  diff vs explicit head={wrap_diff * 1e3:.3f} meV  "
              f"{'OK' if wrapper_ok else 'FAIL'}")
    except Exception as exc:  # noqa: BLE001 - 报清楚而不是抛栈
        print(f"HarmonicallyForcedDP : ERROR: {exc.__class__.__name__}: {exc}")
    all_ok &= wrapper_ok

    # --- 时延（扰动坐标以绕开 ASE 结果缓存）---
    try:
        rng = np.random.default_rng(0)
        frame = slab.copy()
        frame.calc = patched_dp(model=model_used)
        frame.get_potential_energy()  # warmup（含首次加载）
        timings = []
        for _ in range(max(1, repeats)):
            pos = frame.get_positions()
            pos[:, 0] += rng.normal(0.0, 0.005, len(pos))
            frame.set_positions(pos)
            t0 = __import__("time").perf_counter()
            frame.get_potential_energy(), frame.get_forces()
            timings.append(__import__("time").perf_counter() - t0)
        print(f"timing: {len(slab)}-atom Fe(110) slab, median {np.median(timings) * 1e3:.1f} ms / E+F eval")
    except Exception as exc:  # noqa: BLE001 - 报清楚而不是抛栈
        all_ok = False
        print(f"timing: ERROR: {exc.__class__.__name__}: {exc}")

    print("")
    print("result:", "PASS" if all_ok else "FAIL")
    return all_ok
