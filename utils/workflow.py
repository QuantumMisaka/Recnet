from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_legacy_workflow_class():
    legacy_path = Path(__file__).resolve().parents[2] / "run_dp_ts.py"
    spec = importlib.util.spec_from_file_location("wg_legacy_run_dp_ts", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy workflow module: {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return module.DPWorkflow
    except AttributeError as exc:
        raise ImportError(
            f"Legacy workflow module does not define DPWorkflow: {legacy_path}"
        ) from exc


DPWorkflow = _load_legacy_workflow_class()

__all__ = ["DPWorkflow"]
