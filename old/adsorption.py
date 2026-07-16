import os
import numpy as np
from ase.io import read, write
from ase.optimize import MDMin, QuasiNewton
from deepmd.calculator import DP
from utils import constraints as constraint_utils
from utils import geometry as geom
from workflow.utils.helpers import (
    build_constraints, site_lift_vector, is_adsorbate_structure_intact,
    is_adsorbate_close_to_surface, energy_from_xyz
)

class AdsorptionHandler:
    def __init__(self, context):
        self.ctx = context
        self.model = context.MODEL  # 从全局常量或 context 中获取

    def run(self):
        slab = self.ctx.slab
        os.makedirs(os.path.join(self.ctx.path, "rxn", "Adsorbates"), exist_ok=True)

        self.ctx.valid_sites = {}
        self.ctx.valid_sites_all = {}

        for sp_id, data in self.ctx.ads_templates.items():
            ads_dir = self._adsorbate_dir_candidates(sp_id)[0]
            os.makedirs(ads_dir, exist_ok=True)
            sites = slab.unique_sites["idx"]
            myslab = slab.stru.copy()
            energy = {}

            for site in sites:
                path = ads_dir
                prefix = f"{site}"
                tagged_prefix = self.ctx._tagged_stem(prefix)
                pos = slab.get_site(site)
                stru = data["atoms"].copy()

                site_bond_params_list, center_bond_params_list = self._build_site_and_center_anchors(
                    data, pos, len(myslab)
                )

                if self.ctx.enable_rotation_enum_ads:
                    stru, best_angle, best_energy = self._enumerate_azimuth_orientation(
                        base_stru=stru,
                        ads_idx=data["ad_idx"],
                        myslab=myslab,
                        pos=pos,
                        site_bond_params_list=site_bond_params_list,
                        center_bond_params_list=center_bond_params_list,
                        atom_bond_params_list=[],
                    )
                    if best_energy is not None:
                        print(f"Ads enum {sp_id} site {site}: angle={best_angle:.1f} deg, E={best_energy:.6f} eV")

                stru.translate(np.array(pos, dtype=float) + site_lift_vector(self.ctx.surface_normal, self.ctx.adsorbate_lift))
                ads = myslab + stru
                write(os.path.join(path, f"{tagged_prefix}.xyz"), ads)

                e_site = self._optimize_adsorbate(
                    path, tagged_prefix, site_bond_params_list, [],
                    len(myslab), center_bond_params_list
                )
                if e_site is not None:
                    opt_xyz = os.path.join(path, f"{tagged_prefix}_opt.xyz")
                    if os.path.exists(opt_xyz):
                        intact, reason = is_adsorbate_structure_intact(
                            opt_xyz, sp_id, self.ctx.ads_templates, len(myslab)
                        )
                        if not intact:
                            print(f"Reject site {site} for {sp_id}: dissociated ({reason})")
                            e_site = None
                        else:
                            opt_atoms = read(opt_xyz)
                            close_ok, height = is_adsorbate_close_to_surface(
                                opt_atoms, sp_id, pos, self.ctx.surface_normal, len(myslab), max_height=2
                            )
                            if not close_ok:
                                print(f"Site {site} for {sp_id} too far (height={height:.3f} A). Rescue.")
                                e_site = self._rescue_adsorbate(
                                    opt_atoms, data, pos, len(myslab), sp_id, site
                                )
                energy[site] = e_site

            # 筛选有效位点
            valid_energy = {k: v for k, v in energy.items() if v is not None}
            if valid_energy:
                valid_sites = sorted(valid_energy, key=valid_energy.get)
            else:
                # fallback 逻辑
                valid_sites = self._fallback_sites(ads_dir, sites, len(myslab), sp_id)
            self.ctx.valid_sites_all[sp_id] = list(valid_sites)
            if self.ctx.top_x and self.ctx.top_x > 0:
                self.ctx.valid_sites[sp_id] = valid_sites[:self.ctx.top_x]
            else:
                self.ctx.valid_sites[sp_id] = valid_sites
            print(f"Valid sites for {sp_id}: {self.ctx.valid_sites[sp_id]}")

        # 更新反应的有效位点
        for rxn in self.ctx.rxns_dict:
            rxn["valid_reactant_sites"] = []
            rxn["valid_reactant_sites_all"] = []
            for sp_id in rxn["reactant_species"]:
                rxn["valid_reactant_sites"] = self.ctx.valid_sites.get(sp_id, self.ctx.slab.unique_sites["idx"])
                rxn["valid_reactant_sites_all"] = self.ctx.valid_sites_all.get(sp_id, self.ctx.slab.unique_sites["idx"])
            # 产品位点根据需要也可设置，但当前逻辑未使用

    # ---------- 内部方法（迁移自原类） ----------
    def _adsorbate_dir_candidates(self, sp_id):
        # 复制原方法
        pass

    def _build_site_and_center_anchors(self, template, site_pos, slab_atom_count):
        # 复制原方法
        pass

    def _enumerate_azimuth_orientation(self, base_stru, ads_idx, myslab, pos,
                                       site_bond_params_list, center_bond_params_list,
                                       atom_bond_params_list=None):
        # 复制原方法
        pass

    def _optimize_adsorbate(self, path, prefix, site_bond_params_list, atom_bond_params_list,
                            surf_atom_num, center_bond_params_list=None):
        # 复制原 opt 函数逻辑
        pass

    def _rescue_adsorbate(self, atoms, data, pos, slab_atom_count, sp_id, site):
        # 复制救援优化逻辑
        pass

    def _fallback_sites(self, ads_dir, sites, slab_atom_count, sp_id):
        # 复制 fallback 逻辑
        pass