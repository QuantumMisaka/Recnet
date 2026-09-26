#!/usr/bin/env python3
"""Evaluate one geometry with both arms and compare against each arm's relaxed CO*.

Usage: eval_geometry_both.py <geometry.xyz|STRU> <reference.xyz>
Prints dE = E(geometry) - E(reference) for S and M at the same cell/composition.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from deepmd.calculator import DP

R = Path("/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva")
MODELS = {
    "S": R / "models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-single100k-v20260919/checkpoints/model.ckpt-100000.pt",
    "M": R / "models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-multi200k-v20260920/checkpoints/model.ckpt-200000.pt",
}
CASE = {"S": R / "recnet-runs/campaign-c2/S-c3", "M": R / "recnet-runs/campaign-c2/M-c3"}


def energy(model, head, atoms):
    a = atoms.copy()
    a.calc = DP(model=str(model), head=head)
    return float(a.get_potential_energy())


def main() -> int:
    geo = sys.argv[1]
    ref = sys.argv[2]
    src = Path(geo)
    try:
        atoms = read(src, format="abacus")
    except Exception:
        atoms = read(src)
    sym = atoms.get_chemical_symbols()
    ci = max(i for i, s in enumerate(sym) if s == "C")
    oi = max(i for i, s in enumerate(sym) if s == "O")
    d = float(np.linalg.norm(atoms.positions[ci] - atoms.positions[oi]))
    print(f"geometry: {src}  d(C-O) = {d:.3f} Å")
    ref_atoms = read(Path(ref))
    for arm, model in MODELS.items():
        e_geo = energy(model, "ft2dp", atoms)
        e_ref = energy(model, "ft2dp", ref_atoms)
        print(f"{arm}: E(geom)={e_geo:.4f}  E(ref={Path(ref).name})={e_ref:.4f}  "
              f"dE = {e_geo - e_ref:+.3f} eV  ({'model also goes downhill' if e_geo < e_ref else 'model goes uphill'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
