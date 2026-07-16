import os
import yaml
import hashlib
import numpy as np
from ase.io import read, write
from ase.optimize import QuasiNewton
from deepmd.calculator import DP
from workflow.utils.helpers import (
    build_constraints, energy_from_xyz, thermo_analysis, gas_ideal_thermo_corrections,
    is_adsorbate_structure_intact, adsorbate_bond_set
)

class EnergyHandler:
    def __init__(self, context):
        self.ctx = context
        self.energy_cache = {}
        self.vib_cache = {}

    def run_final_state_energy(self):
        records = self.ctx.ts_records
        if not records:
            print("No TS records; skip FS energy.")
            return []

        out_root = os.path.join(self.ctx.path, "rxn", "FS_energy")
        os.makedirs(out_root, exist_ok=True)

        # 优化 slab
        slab_opt_xyz = os.path.join(out_root, self.ctx._tagged_stem("slab_opt") + ".xyz")
        e_slab, slab_atoms = self._get_slab_energy(slab_opt_xyz)
        slab_vib = self._vib_corrections_for_structure(slab_opt_xyz)

        results = []
        for rec in records:
            tag = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
            rxn_idx = self.ctx._resolve_record_rxn_index(rec)
            if rxn_idx is None:
                continue
            rxn = self.ctx.rxns_dict[rxn_idx]
            site = int(rec["site"])

            # 获取 IS 能量 (reactant)
            is_items = self._collect_state_items(rxn["reactant_species"], site, tag, "IS", far_separated=True)
            if is_items is None:
                continue
            fs_items = self._collect_state_items(rxn["product_species"][:2], site, tag, "FS", far_separated=True)
            if fs_items is None:
                continue

            ts_xyz = rec.get("ts_xyz")
            if not ts_xyz or not os.path.exists(ts_xyz):
                continue
            e_ts = self._energy_from_xyz(ts_xyz)
            ts_vib = self._vib_corrections_for_structure(ts_xyz)

            e_is = self._state_energy_from_items(is_items, e_slab)
            e_fs = self._state_energy_from_items(fs_items, e_slab)

            zpe_is = self._state_correction_from_items(is_items, "zpe", slab_vib["zpe"])
            zpe_fs = self._state_correction_from_items(fs_items, "zpe", slab_vib["zpe"])
            zpe_ts = ts_vib["zpe"]

            g_is = self._state_correction_from_items(is_items, "g_corr", slab_vib["g_corr"])
            g_fs = self._state_correction_from_items(fs_items, "g_corr", slab_vib["g_corr"])
            g_ts = ts_vib["g_corr"]

            # 计算各种能量差
            row = {
                "tag": tag,
                "e_is": e_is,
                "e_ts": e_ts,
                "e_fs": e_fs,
                "zpe_is": zpe_is,
                "zpe_ts": zpe_ts,
                "zpe_fs": zpe_fs,
                "g_is": g_is,
                "g_ts": g_ts,
                "g_fs": g_fs,
                # 更多...
            }
            results.append(row)

        # 保存汇总
        results_path = os.path.join(out_root, self.ctx._tagged_stem("final_state_energy") + ".yaml")
        with open(results_path, "w") as f:
            yaml.safe_dump({"final_state_energy": results}, f)
        return results

    def run_adsorption_energy(self):
        out_root = os.path.join(self.ctx.path, "rxn", "FS_energy")
        os.makedirs(out_root, exist_ok=True)

        slab_opt_xyz = os.path.join(out_root, self.ctx._tagged_stem("slab_opt") + ".xyz")
        e_slab, slab_atoms = self._get_slab_energy(slab_opt_xyz)
        slab_vib = self._vib_corrections_for_structure(slab_opt_xyz)

        # 确定气相物种
        gas_species = self._get_gas_species_list()
        results = []
        for sp_id in gas_species:
            ads_xyz = self._find_adsorbate_energy_file_global_min(sp_id)
            if ads_xyz is None:
                continue
            e_ads = self._energy_from_xyz(ads_xyz)
            e_gas = self._energy_from_gas_species(sp_id)
            ads_vib = self._vib_corrections_for_structure(ads_xyz)
            gas_vib = self._vib_corrections_for_gas_species(sp_id)

            delta_e = e_ads - e_slab - e_gas
            delta_zpe = ads_vib["zpe"] - slab_vib["zpe"] - gas_vib["zpe"]
            delta_g = ads_vib["g_corr"] - slab_vib["g_corr"] - gas_vib["g_corr"]
            results.append({
                "species": sp_id,
                "adsorption_energy": delta_e,
                "adsorption_energy_zpe": delta_e + delta_zpe,
                "adsorption_free_energy": delta_e + delta_g,
                "adsorption_free_energy_zpe": delta_e + delta_zpe + delta_g,
            })

        # 保存
        results_path = os.path.join(out_root, self.ctx._tagged_stem("adsorption_energy") + ".yaml")
        with open(results_path, "w") as f:
            yaml.safe_dump({"adsorption_energy": results}, f)
        return results

    # ---------- 内部辅助方法 ----------
    def _get_slab_energy(self, slab_opt_xyz):
        if os.path.exists(slab_opt_xyz):
            atoms = read(slab_opt_xyz)
            atoms.calc = DP(model=MODEL)
            atoms.set_constraint(build_constraints(atoms, self.ctx.normal_axis,
                                                   self.ctx.bottom_freeze_threshold, with_internal_bonds=False))
            e = atoms.get_potential_energy()
        else:
            atoms = self.ctx.slab.stru.copy()
            atoms.calc = DP(model=MODEL)
            atoms.set_constraint(build_constraints(atoms, self.ctx.normal_axis,
                                                   self.ctx.bottom_freeze_threshold, with_internal_bonds=False))
            opt = QuasiNewton(atoms)
            opt.run(fmax=0.05, steps=100)
            e = atoms.get_potential_energy()
            write(slab_opt_xyz, atoms)
        return float(e), atoms

    def _energy_from_xyz(self, xyz_path):
        if xyz_path in self.energy_cache:
            return self.energy_cache[xyz_path]
        e = energy_from_xyz(xyz_path, MODEL, self.ctx.normal_axis, self.ctx.bottom_freeze_threshold)
        self.energy_cache[xyz_path] = e
        return e

    def _energy_from_gas_species(self, sp_id):
        key = f"gas::{sp_id}"
        if key in self.energy_cache:
            return self.energy_cache[key]
        atoms = self.ctx.ads_templates[sp_id]["atoms"].copy()
        atoms.set_cell([20.0, 20.0, 20.0])
        atoms.set_pbc([False, False, False])
        atoms.center()
        atoms.calc = DP(model=MODEL)
        e = float(atoms.get_potential_energy())
        self.energy_cache[key] = e
        return e

    def _vib_corrections_for_structure(self, xyz_path):
        # 实现缓存和调用 thermo_analysis
        pass

    def _vib_corrections_for_gas_species(self, sp_id):
        # 调用 gas_ideal_thermo_corrections
        pass

    def _collect_state_items(self, species_ids, site, tag, state_name, far_separated=False):
        # 构建状态项列表
        pass

    def _state_energy_from_items(self, items, e_slab):
        # 计算多吸附物状态能量
        pass

    def _state_correction_from_items(self, items, key, slab_corr):
        pass

    def _get_gas_species_list(self):
        # 返回需要计算吸附能的物种列表
        pass

    def _find_adsorbate_energy_file_global_min(self, sp_id):
        # 扫描所有位点取能量最低的优化结构
        pass