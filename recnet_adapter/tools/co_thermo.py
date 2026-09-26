#!/usr/bin/env python3
"""CO-dissociation thermodynamics on the model PES (v2, 2026-09-21).

Per (arm, campaign case): take the pipeline's relaxed adsorbate structures and
compare the *best* adsorption site of each species:

    dE = min_site E(C*) + min_site E(O*) - min_site E(CO*)

This is the standard thermodynamic driving force for CO* -> C* + O* on the same
slab/model; a strongly positive value explains why no first-order C-O saddle was
accepted by the pipeline QC.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ase.io import read
from deepmd.calculator import DP

R = Path("/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva")
MODELS = {
    "S": R / "models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-single100k-v20260919/checkpoints/model.ckpt-100000.pt",
    "M": R / "models/ft2dp-v2.2/FT2DPv2.2-dpa4-air-zbl-multi200k-v20260920/checkpoints/model.ckpt-200000.pt",
}
CASES = {arm: [R / f"recnet-runs/campaign-c2/{arm}-c{i}" for i in range(4)] for arm in ("S", "M")}


def eval_energy(model, path: Path):
    atoms = read(path)
    atoms.calc = DP(model=str(model), head="ft2dp")
    return float(atoms.get_potential_energy())


def slab_energy(model, case: Path, vg: int, cache: dict):
    key = (str(case), vg)
    if key in cache:
        return cache[key]
    f = case / f"rxn/FS_energy/slab_opt_vg{vg}.xyz"
    if not f.exists():
        cands = sorted(case.glob(f"rxn/FS_energy/slab_opt_vg{vg}*.xyz"))
        f = cands[0] if cands else None
    if f is None:
        raise FileNotFoundError(f"slab reference missing for vg{vg} in {case}")
    e = eval_energy(model, f)
    cache[key] = e
    return e


def best_site(model, case: Path, prefix: str):
    best = None
    for d in sorted(case.glob(f"rxn/Adsorbates/{prefix}*")):
        for f in sorted(d.glob("*_opt.xyz")):
            m = re.match(r"(\d+)_vg(\d+)_opt\.xyz", f.name)
            if not m:
                continue
            try:
                e = eval_energy(model, f)
            except Exception as exc:
                print(f"  ! {f.name}: {type(exc).__name__}: {exc}")
                continue
            if best is None or e < best[0]:
                best = (e, int(m.group(1)), int(m.group(2)), f)
    return best


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/co_thermo2")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for arm, cases in CASES.items():
        model = MODELS[arm]
        for case in cases:
            if not case.is_dir():
                print(f"skip {case} (missing)")
                continue
            bc = best_site(model, case, "co_")
            bc_ = best_site(model, case, "c_")
            bo = best_site(model, case, "o_")
            if not (bc and bc_ and bo):
                print(f"{arm} {case.name}: incomplete (co={bool(bc)} c={bool(bc_)} o={bool(bo)})")
                continue
            cache = {}
            e_ads_co = bc[0] - slab_energy(model, case, bc[2], cache)
            e_ads_c = bc_[0] - slab_energy(model, case, bc_[2], cache)
            e_ads_o = bo[0] - slab_energy(model, case, bo[2], cache)
            de = e_ads_c + e_ads_o - e_ads_co
            rec = {"arm": arm, "case": case.name,
                   "E_ads_CO": e_ads_co, "site_CO": bc[1], "vg_CO": bc[2],
                   "E_ads_C": e_ads_c, "site_C": bc_[1], "vg_C": bc_[2],
                   "E_ads_O": e_ads_o, "site_O": bo[1], "vg_O": bo[2],
                   "dE_diss": de}
            results.append(rec)
            print(f"{arm} {case.name}: dE(C+O - CO) = {de:+.3f} eV "
                  f"(CO@site{bc[1]}/vg{bc[2]}, C@site{bc_[1]}, O@site{bo[1]})", flush=True)
    (out / "co_thermo.json").write_text(json.dumps({"results": results}, indent=1))
    if results:
        v = sorted(r["dE_diss"] for r in results)
        print(f"\nsummary: n={len(v)} min={v[0]:+.3f} median={v[len(v)//2]:+.3f} max={v[-1]:+.3f} eV")
    print("wrote", out / "co_thermo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
