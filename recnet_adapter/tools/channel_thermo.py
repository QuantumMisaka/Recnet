#!/usr/bin/env python3
"""Thermodynamic characterisation of the remaining boundary channels (2026-09-21).

Same convention as ``co_thermo.py``: every species is evaluated at its own best
adsorbed site (pipeline-relaxed structures in ``rxn/Adsorbates/<species>/``), and
adsorption energies are referenced to the matching ``rxn/FS_energy/slab_opt_vg<k>.xyz``.

Channels (reactant -> products):
    CH*   -> C* + H*          dE = Eads(C) + Eads(H) - Eads(CH)
    CCO*  -> CC* + O*         dE = Eads(CC) + Eads(O) - Eads(CCO)
    CCO*  -> C* + CO*         dE = Eads(C) + Eads(CO) - Eads(CCO)
    CHCO* -> CCH* + O*        dE = Eads(CCH) + Eads(O) - Eads(CHCO)
    H2(g) -> H* + H*          dE = 2*Eads(H) - E(H2_gas)      (gas reference from the H2 template)

Writes ``channel_thermo.json`` in the output dir.
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
CASES = {
    arm: ([R / f"recnet-runs/campaign-c2/{arm}-c{i}" for i in range(4)]
          + [R / f"recnet-runs/targeted-r6/{tag}-{arm}" for tag in ("ch", "co", "chco")])
    for arm in ("S", "M")
}
CHANNELS = [
    ("CH", ["C", "H"], ["CH"]),
    ("CCO", ["CC", "O"], ["CCO"]),
    ("CCO", ["C", "CO"], ["CCO"]),
    ("CHCO", ["CCH", "O"], ["CHCO"]),
]
SPECIES_PREFIX = {"C": "c_", "H": "h_", "CH": "ch_", "CC": "cc_", "CO": "co_",
                  "O": "o_", "CCO": "cco_", "CCH": "cch_", "CHCO": "chco_"}


def energy(model, path):
    a = read(path)
    a.calc = DP(model=str(model), head="ft2dp")
    return float(a.get_potential_energy())


def best_site(model, case: Path, prefix: str, cache: dict):
    key = (str(case), prefix)
    if key in cache:
        return cache[key]
    best = None
    for d in sorted(case.glob(f"rxn/Adsorbates/{prefix}*")):
        for f in sorted(d.glob("*_opt.xyz")):
            m = re.match(r"(\d+)_vg(\d+)_opt\.xyz", f.name)
            if not m:
                continue
            try:
                e = energy(model, f)
            except Exception:
                continue
            if best is None or e < best[0]:
                best = (e, int(m.group(1)), int(m.group(2)), f)
    cache[key] = best
    return best


def slab_energy(model, case: Path, vg: int, cache: dict):
    key = (str(case), "slab", vg)
    if key in cache:
        return cache[key]
    f = case / f"rxn/FS_energy/slab_opt_vg{vg}.xyz"
    if not f.exists():
        cands = sorted(case.glob(f"rxn/FS_energy/slab_opt_vg{vg}*.xyz"))
        f = cands[0] if cands else None
    if f is None:
        return None
    e = energy(model, f)
    cache[key] = e
    return e


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/channel_thermo")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    gas_cache = {}
    for arm, cases in CASES.items():
        model = MODELS[arm]
        cache = {}
        for case in cases:
            if not case.is_dir():
                continue
            vals = {}
            ok = True
            for ch, prods, reactants in CHANNELS:
                items = {}
                for sp in set(prods + reactants):
                    pref = SPECIES_PREFIX[sp]
                    b = best_site(model, case, pref, cache)
                    if b is None:
                        ok = False
                        break
                    slab = slab_energy(model, case, b[2], cache)
                    if slab is None:
                        ok = False
                        break
                    items[sp] = (b[0] - slab, b[1], b[2])
                if not ok:
                    continue
                d_e = sum(items[p][0] for p in prods) - sum(items[r][0] for r in reactants)
                rows.append({"arm": arm, "case": case.name, "channel": f"{ch}* -> {' + '.join(p + '*' for p in prods)}",
                             "dE_eV": round(d_e, 4), "sites": {k: (v[1], v[2]) for k, v in items.items()}})
                print(f"{arm} {case.name:6s} {ch}* -> {'+'.join(prods):8s} dE = {d_e:+.3f} eV", flush=True)
            # H2(g) -> 2H*
            b = best_site(model, case, "h_", cache)
            slab = slab_energy(model, case, b[2], cache) if b else None
            if b and slab is not None:
                key = (str(case), "h2gas")
                if key not in gas_cache:
                    # locate the H2 species *by name* in this case's prepared data; its
                    # template is the isolated molecule (ad_idx == []).  A glob fallback
                    # previously picked arbitrary templates (slab-sized!) and produced
                    # nonsense gas references.
                    gas_cache[key] = None
                    try:
                        import yaml
                        prep = sorted(case.glob("prepared_data/*prepared_rmg_data.yaml"))[0]
                        payload = yaml.safe_load(prep.read_text())
                        for sp_id, rec in payload["species"].items():
                            if rec.get("name", "").upper() == "H2":
                                tpl = case / "prepared_data" / rec["template_xyz"]
                                if tpl.exists():
                                    gas_cache[key] = energy(model, tpl)
                                break
                    except Exception:
                        pass
                e_gas = gas_cache[key]
                if e_gas is not None:
                    d_e = 2 * (b[0] - slab) - e_gas
                    rows.append({"arm": arm, "case": case.name, "channel": "H2(g) -> H* + H*",
                                 "dE_eV": round(d_e, 4), "sites": {"H": (b[1], b[2])}})
                    print(f"{arm} {case.name:6s} H2(g) -> 2H*    dE = {d_e:+.3f} eV (gas ref {e_gas:.2f})", flush=True)
    (out / "channel_thermo.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {out/'channel_thermo.json'} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
