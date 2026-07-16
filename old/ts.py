import os
import numpy as np
import shutil
import yaml
from ase.io import read, write
from ase.optimize import MDMin, QuasiNewton
from sella import Sella
from deepmd.calculator import DP
from utils import constraints as constraint_utils
from utils import geometry as geom
from ccqn import CCQN
from workflow.utils.helpers import (
    build_constraints, site_lift_vector, is_ts_bond_overstretched,
    ts_bond_dissociation_threshold, get_imag_mode_report,
    displace_and_relax_imag_mode, significant_imag_frequencies,
    orient_broken_bond_torsion, add_reactive_endpoint_site_anchor
)

class TSHandler:
    def __init__(self, context):
        self.ctx = context

    def run(self):
        slab = self.ctx.slab
        os.makedirs(os.path.join(self.ctx.path, "rxn", "TS_guesses"), exist_ok=True)

        # 加载已有记录
        old_records = self.ctx.ts_records
        records_by_tag = {rec.get("tag"): rec for rec in old_records if isinstance(rec, dict) and rec.get("tag")}
        self.ctx.ts_records = list(records_by_tag.values())

        for idx, rxn in enumerate(self.ctx.rxns_dict):
            preferred_sites = list(rxn.get("valid_reactant_sites", []))
            all_sites = list(rxn.get("valid_reactant_sites_all", preferred_sites))
            sites = self._dedupe_keep_order(preferred_sites + [s for s in all_sites if s not in preferred_sites])
            backup_sites = set(s for s in sites if s not in preferred_sites)
            accepted_any_site = False
            rec_bond = rxn["broken_bond"]
            sp_id = rxn["reactant_species"][0]
            rxn_key = self.ctx.rxn_key_by_index[idx]
            myslab = slab.stru.copy()
            template = self.ctx.ads_templates[sp_id]

            for site in sites:
                if site in backup_sites and accepted_any_site:
                    continue
                try:
                    ts_stem = self.ctx._tagged_stem(f"{rxn_key}_site_{site}")
                    ts_tag = ts_stem
                    # 路径定义
                    out_xyz = os.path.join(self.ctx.path, "rxn", "TS_guesses", f"{ts_stem}_opt_ccqn.xyz")
                    ts_opt_xyz = os.path.join(self.ctx.path, "rxn", "TS_guesses", f"{ts_stem}_opt.xyz")
                    ts_guess_xyz = os.path.join(self.ctx.path, "rxn", "TS_guesses", f"{ts_stem}.xyz")
                    summary_log = self.ctx._tagged_stem(f"optimization_summary_{rxn_key}_site_{site}") + ".log"

                    # 如果已有结果则跳过
                    if os.path.exists(out_xyz):
                        print(f"Reuse existing TS result for {ts_stem}")
                        self._save_ts_record_if_needed(ts_tag, idx, site, sp_id, rxn_key, rec_bond,
                                                       len(myslab), ts_opt_xyz, out_xyz, summary_log)
                        continue

                    pos = slab.get_site(site)
                    print(f"Processing rxn_{idx} at site {site} with rec_bond {rec_bond}")

                    # 准备初始结构
                    stru_seed = template["atoms"].copy()
                    # 预处理断键方向
                    orient_broken_bond_torsion(stru_seed, rec_bond, self.ctx.surface_normal, template["ad_idx"])
                    # 内角旋转
                    # (调用 geom.rotate_about_ads_inner_angle)
                    # 方位角枚举
                    azimuth_candidates = self._get_azimuth_candidates(stru_seed, template, myslab, pos)

                    accepted = False
                    attempt_msgs = []
                    for az_angle in azimuth_candidates:
                        # 执行 CCQN 优化并检查虚频
                        success, msg = self._attempt_ts_optimization(
                            stru_seed, template, pos, myslab, rec_bond, site,
                            ts_tag, az_angle, idx, sp_id, rxn_key
                        )
                        attempt_msgs.append(msg)
                        if success:
                            accepted = True
                            break

                    # 写日志
                    with open(summary_log, "w") as f:
                        f.write("\n".join(attempt_msgs))

                    if accepted:
                        accepted_any_site = True
                        self._save_ts_record_if_needed(ts_tag, idx, site, sp_id, rxn_key, rec_bond,
                                                       len(myslab), ts_opt_xyz, out_xyz, summary_log)
                except Exception as e:
                    print(f"Error for rxn_{idx}_site_{site}: {e}")
                    continue

        self.ctx._flush_ts_records()

    # ---------- 内部方法 ----------
    def _attempt_ts_optimization(self, stru_seed, template, pos, myslab, rec_bond, site,
                                 ts_tag, az_angle, rxn_idx, sp_id, rxn_key):
        # 核心 CCQN 优化 + 虚频检查
        # 返回 (success, message)
        pass

    def _get_azimuth_candidates(self, stru_seed, template, myslab, pos):
        # 从枚举中获取角度列表
        pass

    def _save_ts_record_if_needed(self, tag, rxn_idx, site, sp_id, rxn_key, rec_bond,
                                  slab_atom_count, initial_xyz, ts_xyz, summary_log):
        # 调用 self.ctx._save_ts_record (需实现)
        pass

    def _dedupe_keep_order(self, seq):
        # 可以用 cfgmod._dedupe_keep_order
        pass