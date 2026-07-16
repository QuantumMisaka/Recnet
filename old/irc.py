import os
import yaml
import hashlib
import numpy as np
from ase.io import read, write
from ase.optimize import QuasiNewton
from ase.geometry import get_distances
from ase.data import covalent_radii
from deepmd.calculator import DP
from sella import IRC
from workflow.utils.helpers import build_constraints, thermo_analysis

class IRCHandler:
    def __init__(self, context):
        self.ctx = context

    def run(self):
        records = self.ctx.ts_records
        if not records:
            print("No TS records; skip IRC.")
            return []

        out_root = os.path.join(self.ctx.path, "rxn", "IRC_final_states")
        os.makedirs(out_root, exist_ok=True)

        results = []
        for rec in records:
            tag = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
            case_dir = os.path.join(out_root, tag)
            os.makedirs(case_dir, exist_ok=True)

            ts_atoms = read(rec["ts_xyz"])
            init_atoms = read(rec["initial_xyz"])
            rec_i, rec_j = rec["rec_bond_global"]

            # 设置计算器
            ts_atoms.calc = DP(model=MODEL)
            ts_atoms.set_constraint(build_constraints(ts_atoms, self.ctx.normal_axis,
                                                      self.ctx.bottom_freeze_threshold, with_internal_bonds=False))
            init_atoms.calc = DP(model=MODEL)
            init_atoms.set_constraint(build_constraints(init_atoms, self.ctx.normal_axis,
                                                        self.ctx.bottom_freeze_threshold, with_internal_bonds=False))

            # 运行 IRC forward & reverse
            forward_traj = os.path.join(case_dir, "irc_forward.traj")
            reverse_traj = os.path.join(case_dir, "irc_reverse.traj")

            irc_fwd = IRC(ts_atoms.copy(), trajectory=forward_traj,
                          dx=self.ctx.irc_dx, eta=self.ctx.irc_eta,
                          ninner_iter=self.ctx.irc_ninner_iter,
                          logfile=os.path.join(case_dir, "irc_forward.log"))
            irc_fwd.run(fmax=self.ctx.irc_fmax, steps=self.ctx.irc_steps, direction="forward")

            irc_rev = IRC(ts_atoms.copy(), trajectory=reverse_traj,
                          dx=self.ctx.irc_dx, eta=self.ctx.irc_eta,
                          ninner_iter=self.ctx.irc_ninner_iter,
                          logfile=os.path.join(case_dir, "irc_reverse.log"))
            irc_rev.run(fmax=self.ctx.irc_fmax, steps=self.ctx.irc_steps, direction="reverse")

            # 收集路径并选择最终态
            forward_path = read(forward_traj, index=":")
            reverse_path = read(reverse_traj, index=":")
            if not forward_path or not reverse_path:
                print(f"IRC path empty for {tag}")
                continue

            forward_final = forward_path[-1]
            reverse_final = reverse_path[-1]

            # 计算断键距离
            _, dmat = get_distances(init_atoms.get_positions(), None, init_atoms.get_cell(), [True, True, True])
            d_init = dmat[rec_i, rec_j]
            _, dmat_f = get_distances(forward_final.get_positions(), None, forward_final.get_cell(), [True, True, True])
            d_f = dmat_f[rec_i, rec_j]
            _, dmat_r = get_distances(reverse_final.get_positions(), None, reverse_final.get_cell(), [True, True, True])
            d_r = dmat_r[rec_i, rec_j]

            # 选择解离更明显的分支
            chosen_branch = "forward" if d_f >= d_r else "reverse"
            chosen_final = forward_final if chosen_branch == "forward" else reverse_final
            chosen_d = d_f if chosen_branch == "forward" else d_r

            # 若解离不充分，尝试使用虚频扰动端点
            dissoc_threshold = max(1.35 * d_init, 1.25 * (covalent_radii[ts_atoms[rec_i].number] + covalent_radii[ts_atoms[rec_j].number]), d_init + 0.35)
            if chosen_d < dissoc_threshold:
                chosen_final, chosen_branch, chosen_d = self._try_perturbation_fallback(
                    tag, rec, rec_i, rec_j, dissoc_threshold, chosen_d, case_dir
                )

            # 热力学修正
            vib_corr = {}
            if self.ctx.enable_irc_thermo_corrections:
                for label, atoms in [("initial", init_atoms), ("ts", ts_atoms), ("selected", chosen_final)]:
                    vib_name = os.path.join(case_dir, f"{label}_vib")
                    g_corr, zpe, _ = thermo_analysis(atoms, self.ctx.temperature, name=vib_name,
                                                      normal_axis=self.ctx.normal_axis,
                                                      slab_atom_count=len(self.ctx.slab.stru))
                    vib_corr[label] = {"g_corr": g_corr, "zpe": zpe}
            else:
                vib_corr = {k: {"g_corr": 0.0, "zpe": 0.0} for k in ["initial", "ts", "selected"]}

            # 写入结果
            result = {
                "tag": tag,
                "chosen_branch": chosen_branch,
                "d_break_initial": d_init,
                "d_break_selected": chosen_d,
                "e_initial": init_atoms.get_potential_energy(),
                "e_ts": ts_atoms.get_potential_energy(),
                "e_selected": chosen_final.get_potential_energy(),
                "zpe_initial": vib_corr["initial"]["zpe"],
                "zpe_ts": vib_corr["ts"]["zpe"],
                "zpe_selected": vib_corr["selected"]["zpe"],
                "g_initial": vib_corr["initial"]["g_corr"],
                "g_ts": vib_corr["ts"]["g_corr"],
                "g_selected": vib_corr["selected"]["g_corr"],
                # 更多计算...
            }
            results.append(result)
            with open(os.path.join(case_dir, "irc_summary.yaml"), "w") as f:
                yaml.safe_dump(result, f)

        # 保存汇总
        with open(os.path.join(out_root, "irc_results.yaml"), "w") as f:
            yaml.safe_dump({"irc_results": results}, f)
        return results

    def _try_perturbation_fallback(self, tag, rec, rec_i, rec_j, dissoc_threshold, current_d, case_dir):
        # 尝试从虚频扰动端点加载结构
        ts_stem = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
        pert_paths = [
            ("plus", os.path.join(self.ctx.path, "rxn", "TS_guesses", f"{ts_stem}_imag_plus.xyz")),
            ("minus", os.path.join(self.ctx.path, "rxn", "TS_guesses", f"{ts_stem}_imag_minus.xyz")),
        ]
        best = None
        for label, path in pert_paths:
            if os.path.exists(path):
                atoms = read(path)
                atoms.calc = DP(model=MODEL)
                _, dmat = get_distances(atoms.get_positions(), None, atoms.get_cell(), [True, True, True])
                d = dmat[rec_i, rec_j]
                if d >= dissoc_threshold and d > current_d:
                    if best is None or d > best[1]:
                        best = (atoms, f"perturbation_{label}", d)
        if best:
            atoms, branch, d = best
            write(os.path.join(case_dir, "final_perturbation.xyz"), atoms)
            return atoms, branch, d
        return None, "forward", current_d  # fallback