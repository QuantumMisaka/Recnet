"""Core-only wheel/editable install gate (no adapter/builder assumptions).

Builds the wheel from this worktree, asserts core members and absence of
fork-only payloads, then probes wheel and editable installs in throwaway
venvs. The invoking interpreter must provide the core/pipeline dependencies
(ase, numpy, scipy, PyYAML, sella, deepmd) via ``--system-site-packages``;
this validates packaging, not a fresh dependency resolution from an index.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "handlers/__init__.py",
    "handlers/context.py",
    "handlers/energy.py",
    "handlers/ts.py",
    "handlers/workflow.py",
    "ccqn/__init__.py",
    "ccqn/ccqn.py",
    "utils/__init__.py",
    "utils/config.py",
    "utils/constraints.py",
    "utils/geometry.py",
    "run_dp_ts.py",
    "slabsite.py",
}
FORBIDDEN_PREFIXES = ("recnet_adapter/", "network_builder/", "network_inputs/")
FORBIDDEN_PARTS = ("/tests/", "/evidence/", "/sai/")


def run(command, cwd, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, timeout=180)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-python",
        help="Also probe installed pipeline imports in this environment",
    )
    args = parser.parse_args()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "DISPLAY"):
        env.pop(key, None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    with tempfile.TemporaryDirectory(prefix="recnet-core-install-") as directory:
        scratch = Path(directory)
        wheel_dir = scratch / "wheels"
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                wheel_dir,
                ROOT,
            ],
            scratch,
            env,
        )
        wheel, = wheel_dir.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        missing = REQUIRED - names
        if missing:
            raise RuntimeError(f"Missing wheel members: {sorted(missing)}")
        bad = sorted(
            name
            for name in names
            if name.startswith(FORBIDDEN_PREFIXES)
            or any(part in name for part in FORBIDDEN_PARTS)
        )
        if bad:
            raise RuntimeError(f"Wheel contains fork-only payloads: {bad}")

        for mode in ("wheel", "editable"):
            venv = scratch / mode
            run(
                [sys.executable, "-m", "venv", "--system-site-packages", venv],
                scratch,
                env,
            )
            python = venv / "bin/python"
            target = [wheel] if mode == "wheel" else ["--editable", ROOT]
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--no-index",
                    *target,
                ],
                scratch,
                env,
            )
            expected = venv if mode == "wheel" else ROOT
            probe = (
                "from pathlib import Path; "
                "import importlib.metadata as metadata; "
                "import handlers, ccqn, utils, slabsite; "
                f"root=Path({str(expected)!r}).resolve(); "
                "assert all(Path(mod.__file__).resolve().is_relative_to(root) "
                "for mod in (handlers, ccqn, utils, slabsite)); "
                "assert metadata.version('recnet') == '0.0.1'; "
                "print('PASS core installed paths and version')"
            )
            run([python, "-c", probe], scratch, env)

        if args.pipeline_python:
            venv = scratch / "pipeline"
            run(
                [args.pipeline_python, "-m", "venv", "--system-site-packages", venv],
                scratch,
                env,
            )
            python = venv / "bin/python"
            run([python, "-m", "pip", "install", "--no-deps", "--no-index", wheel], scratch, env)
            probe = (
                "from pathlib import Path; "
                "import handlers, ccqn, utils, slabsite, run_dp_ts; "
                f"root=Path({str(venv)!r}).resolve(); "
                "assert all(Path(mod.__file__).resolve().is_relative_to(root) "
                "for mod in (handlers, ccqn, utils, slabsite, run_dp_ts)); "
                "print('PASS pipeline imports (no model execution)')"
            )
            run([python, "-c", probe], scratch, env)
    print("PASS core package smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
