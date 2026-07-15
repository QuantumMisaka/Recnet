import argparse
import hashlib
import json
import os
import traceback
import shutil
from dataclasses import dataclass
import yaml
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms, FixInternals
from ase.optimize import MDMin, QuasiNewton
from sella import Sella
from ase.calculators import calculator
from ase.geometry import get_distances
from ase.data import covalent_radii
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
import ase

from deepmd.calculator import DP

from slabsite import SlabSite
from ccqn import CCQN

from utils import config as cfgmod
from utils import geometry as geom
from utils import constraints as constraint_utils

MODEL = "/data/home/youyinglong/model/dpa230-v2-simp/FeCHO-dpa231-v2-7-3heads-100w.pth"

class DPWorkflow:
    def __init__(
        self,
        path,
        prepared_data_file,
        slab_path,
        surface_normal,
        normal_axis=None,        
        top_x=None,
        enable_rotation_enum_ads=False,
        enable_rotation_enum_ts=True,
        enable_imag_mode_check=True,
        imag_mode_displacement=0.15,
        imag_mode_relax_steps=40,
        bottom_freeze_threshold=None,
        run_irc_final_state=False,
        irc_fmax=0.08,
        irc_steps=200,
        irc_dx=0.05,
        irc_eta=1e-4,
        irc_ninner_iter=50,
        enable_irc_thermo_corrections=False,
        irc_temperature=523.15,
        enable_thermo_corrections=True,
        gas_species_whitelist=None,
        gas_pressure_pa=101325.0,
        output_suffix="",
    ):
        self.path = path
        self.prepared_data_file = prepared_data_file
        self.slab_path = slab_path
        self.top_x = top_x
        self.enable_rotation_enum_ads = bool(enable_rotation_enum_ads)
        self.enable_rotation_enum_ts = bool(enable_rotation_enum_ts)
        self.rotation_angle_candidates = [0, 90, 180, 270]
        self.surface_normal = tuple(surface_normal)
        self.adsorbate_lift = 0.8
        self.normal_axis = (normal_axis or cfgmod.infer_normal_axis(self.surface_normal)).lower()
        self.enable_imag_mode_check = bool(enable_imag_mode_check)
        self.imag_mode_displacement = float(imag_mode_displacement)
        self.imag_mode_relax_steps = int(imag_mode_relax_steps)
        self.run_irc_final_state = bool(run_irc_final_state)
        self.irc_fmax = float(irc_fmax)
        self.irc_steps = int(irc_steps)
        self.irc_dx = float(irc_dx)
        self.irc_eta = float(irc_eta)
        self.irc_ninner_iter = int(irc_ninner_iter)
        self.enable_irc_thermo_corrections = bool(enable_irc_thermo_corrections)
        self.temperature = float(irc_temperature)
        self.enable_thermo_corrections = bool(enable_thermo_corrections)
        self.gas_species_whitelist = (
            cfgmod._dedupe_keep_order(list(gas_species_whitelist))
            if gas_species_whitelist is not None
            else None
        )
        self.gas_pressure_pa = float(gas_pressure_pa)
        # TS bond-length QC: reject/redo candidates with unrealistically elongated reactive bond.
        self.ts_bond_max_scale_ref = 1.9
        self.ts_bond_max_scale_covalent = 1.6
        self.ts_bond_max_additive_cap = 1.2
        self.ts_endpoint_dissoc_scale_ref = 1.35
        self.ts_endpoint_dissoc_scale_covalent = 1.25
        self.ts_endpoint_dissoc_additive = 0.35
        self.ts_endpoint_min_span = 0.20
        # TS mode QC: suppress drift-dominated or multi-imaginary false positives.
        self.imag_freq_significant_cutoff = 20.0
        self.ts_primary_mode_min_bond_proj = 0.16
        self.ts_primary_mode_bond_margin = 0.02
        
        self.output_suffix = str(output_suffix or "")
        self.bottom_freeze_threshold = (
            float(bottom_freeze_threshold)
            if bottom_freeze_threshold is not None
            else self._default_bottom_freeze_threshold()
        )
        self.ts_records = []
        ts_records_name = self._tagged_stem("ts_records") + ".yaml"
        self.ts_records_path = os.path.join(self.path, "rxn", "TS_archive", ts_records_name)

        with open(self.prepared_data_file, "r") as f:
            payload = yaml.safe_load(f)
        self.rxns_dict = payload["rxns"]
        self.species = payload["species"]

        # For Fe2C(001), keep the old z-surface construction behavior.
        if self.normal_axis == "z":
            stru = read(slab_path)
            fe_atoms = [atom for atom in stru if atom.symbol == "Fe"]
            if len(fe_atoms) > 0:
                z_max = np.max([atom.position[2] for atom in fe_atoms])
                self.slab = SlabSite(slab_path, element="Fe", z_min=z_max - 0.1)
            else:
                self.slab = SlabSite(slab_path, normal_axis=self.normal_axis, z_min=None)
            self.slab.voronoi(True, ["hollow", "bridge", "top"])
            self.slab.filter_unique_site()
        else:
            self.slab = SlabSite(slab_path, normal_axis=self.normal_axis, z_min=None)
            self.slab.voronoi(False, ["hollow"])
            self.slab.filter_unique_site()
            if len(self.slab.unique_sites["idx"]) == 0:
                self.slab.voronoi(False, ["top", "bridge", "hollow"])
                self.slab.filter_unique_site()
        self.slab.find_pairs(radius=3.5, neighbors_num=5, min_dist=1.5)

        self.ads_templates = {}
        base_dir = os.path.dirname(self.prepared_data_file)
        for sp_id, rec in self.species.items():
            template_path = os.path.join(base_dir, rec["template_xyz"])
            self.ads_templates[sp_id] = {
                "atoms": read(template_path),
                "ad_idx": rec["ad_idx"],
                "name": rec.get("name", sp_id),
            }

        self.species_key_by_id = {}
        self.species_ids_by_key = {}
        for sp_id in self.ads_templates:
            sp_key = self._species_key(sp_id)
            self.species_key_by_id[sp_id] = sp_key
            self.species_ids_by_key.setdefault(sp_key, []).append(sp_id)

        self.rxn_key_by_index = []
        self.rxn_index_by_key = {}
        for rxn_idx, rxn in enumerate(self.rxns_dict):
            rxn_key = self._reaction_key(rxn)
            self.rxn_key_by_index.append(rxn_key)
            if rxn_key not in self.rxn_index_by_key:
                self.rxn_index_by_key[rxn_key] = rxn_idx

    def _tagged_stem(self, stem):
        if self.output_suffix:
            return f"{stem}_{self.output_suffix}"
        return stem

    def _slugify_token(self, text, fallback="item"):
        token = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(text))
        while "__" in token:
            token = token.replace("__", "_")
        token = token.strip("_")
        return token or fallback

    def _stable_digest(self, payload, n=12):
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:n]

    def _species_key(self, sp_id):
        template = self.ads_templates[sp_id]
        atoms = template["atoms"]
        ad_idx = sorted(int(i) for i in template.get("ad_idx", []))

        symbol_counts = {}
        for symbol in atoms.get_chemical_symbols():
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        bonds = []
        for i, j in geom.get_bond_connections(atoms, shift=0, cutoff=1.2, bond_type="nosurf"):
            a, b = int(i), int(j)
            if a > b:
                a, b = b, a
            bonds.append([a, b])
        bonds.sort()

        payload = {
            "name": str(template.get("name", "")).strip().lower(),
            "formula": atoms.get_chemical_formula(mode="hill"),
            "atom_count": int(len(atoms)),
            "ad_idx": ad_idx,
            "symbol_counts": [[k, symbol_counts[k]] for k in sorted(symbol_counts)],
            "bonds": bonds,
        }
        digest = self._stable_digest(payload)
        slug = self._slugify_token(template.get("name", sp_id), fallback="species")
        return f"{slug}_{digest}"

    def _reaction_key(self, rxn):
        def _counted_keys(sp_ids):
            counts = {}
            for sid in sp_ids:
                key = self.species_key_by_id.get(sid, str(sid))
                counts[key] = counts.get(key, 0) + 1
            return [[k, counts[k]] for k in sorted(counts)]

        broken_bond = rxn.get("broken_bond", [])
        bb = []
        if isinstance(broken_bond, (list, tuple)) and len(broken_bond) >= 2:
            a, b = int(broken_bond[0]), int(broken_bond[1])
            bb = [a, b] if a <= b else [b, a]

        payload = {
            "reactants": _counted_keys(rxn.get("reactant_species", [])),
            "products": _counted_keys(rxn.get("product_species", [])),
            "broken_bond": bb,
        }
        return f"rxn_{self._stable_digest(payload)}"

    def _adsorbate_dir_candidates(self, sp_id):
        candidates = []
        sp_key = self.species_key_by_id.get(sp_id)
        if sp_key:
            candidates.append(os.path.join(self.path, "rxn", "Adsorbates", sp_key))
        candidates.append(os.path.join(self.path, "rxn", "Adsorbates", str(sp_id)))

        out = []
        seen = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _resolve_record_rxn_index(self, rec):
        rxn_key = rec.get("rxn_key")
        if rxn_key and rxn_key in self.rxn_index_by_key:
            return int(self.rxn_index_by_key[rxn_key])

        if "rxn_idx" in rec:
            try:
                rxn_idx = int(rec["rxn_idx"])
            except Exception:
                return None
            if 0 <= rxn_idx < len(self.rxns_dict):
                return rxn_idx
        return None

    def _site_lift_vector(self):
        """Return the placement shift along surface normal."""
        return np.array(self.surface_normal, dtype=float) * float(self.adsorbate_lift)

    def _save_ts_record(
        self,
        rxn_idx,
        site,
        sp_id,
        rxn_key,
        rec_bond,
        slab_atom_count,
        initial_xyz,
        ts_xyz,
        summary_log,
        tag=None,
    ):
        """Persist TS info and its corresponding initial structure for later IRC runs."""
        if tag is None:
            tag = self._tagged_stem(f"rxn_{rxn_idx}_site_{site}")
        archive_dir = os.path.join(self.path, "rxn", "TS_archive", tag)
        os.makedirs(archive_dir, exist_ok=True)

        init_copy = os.path.join(archive_dir, "initial.xyz")
        ts_copy = os.path.join(archive_dir, "ts.xyz")
        summary_copy = os.path.join(archive_dir, "optimization_summary.log")

        shutil.copy2(initial_xyz, init_copy)
        shutil.copy2(ts_xyz, ts_copy)
        if os.path.exists(summary_log):
            shutil.copy2(summary_log, summary_copy)

        rec_bond_global = [int(rec_bond[0] + slab_atom_count), int(rec_bond[1] + slab_atom_count)]
        record = {
            "tag": tag,
            "rxn_idx": int(rxn_idx),
            "rxn_key": rxn_key,
            "site": int(site),
            "species": sp_id,
            "species_key": self.species_key_by_id.get(sp_id),
            "rec_bond_local": [int(rec_bond[0]), int(rec_bond[1])],
            "rec_bond_global": rec_bond_global,
            "initial_xyz": os.path.abspath(init_copy),
            "ts_xyz": os.path.abspath(ts_copy),
            "summary_log": os.path.abspath(summary_copy),
        }

        with open(os.path.join(archive_dir, "meta.yaml"), "w") as f:
            yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)

        self.ts_records.append(record)

    def _flush_ts_records(self):
        os.makedirs(os.path.dirname(self.ts_records_path), exist_ok=True)
        with open(self.ts_records_path, "w") as f:
            yaml.safe_dump({"ts_records": self.ts_records}, f, sort_keys=False, allow_unicode=True)

    def _load_ts_records(self):
        if self.ts_records:
            return self.ts_records
        if not os.path.exists(self.ts_records_path):
            return []
        with open(self.ts_records_path, "r") as f:
            payload = yaml.safe_load(f) or {}
        records = payload.get("ts_records", [])
        if not isinstance(records, list):
            return []
        self.ts_records = records
        return self.ts_records

    def _default_bottom_freeze_threshold(self):
        """Choose a legacy-compatible default freeze threshold by surface axis."""
        if self.normal_axis == "y":
            return 1.5
        return 0.1 * 25.35

    def _frozen_indices(self, atoms):
        """Select bottom-layer atoms along configured surface normal axis."""
        axis_id = {"x": 0, "y": 1, "z": 2}[self.normal_axis]
        return [
            atom.index
            for atom in atoms
            if float(atom.position[axis_id]) < float(self.bottom_freeze_threshold)
        ]

    def _build_constraints(self, atoms, surf_atom_num=0, with_internal_bonds=False):
        """Build reusable structure constraints for relaxation stages."""
        constraints = [FixAtoms(indices=self._frozen_indices(atoms))]
        if with_internal_bonds:
            bonds = geom.get_bond_connections(atoms, shift=surf_atom_num, bond_type="nosurf")
            _, distances = get_distances(atoms.get_positions(), None, atoms.get_cell(), [True, True, True])
            new_bonds = []
            for bond in bonds:
                new_bond = [bond[0] + surf_atom_num, bond[1] + surf_atom_num]
                dis = distances[new_bond[0], new_bond[1]]
                new_bonds.append([dis, new_bond])
            constraints.insert(0, FixInternals(bonds=new_bonds))
        return constraints

    def _build_site_and_center_anchors(self, template, site_pos, slab_atom_count):
        site_bond_params_list = []
        center_bond_params_list = []

        ad_idx = list(template["ad_idx"])

        # Multi-anchor adsorbates: keep all adsorption atoms close to the surface,
        # but avoid collapsing them onto exactly the same target point.
        if len(ad_idx) >= 2:
            positions = template["atoms"].get_positions()
            normal = np.array(self.surface_normal, dtype=float)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                normal = np.array(self.surface_normal, dtype=float)
                normal_norm = np.linalg.norm(normal)
            normal /= normal_norm

            primary_idx = ad_idx[0]
            primary_pos = positions[primary_idx]

            site_bond_params_list.append({
                "site_pos": site_pos,
                "ind": primary_idx + slab_atom_count,
                "k": 0.35,
                "deq": 0.0,
            })

            for idx in ad_idx[1:]:
                rel = positions[idx] - primary_pos
                rel_plane = rel - np.dot(rel, normal) * normal
                if np.linalg.norm(rel_plane) < 1e-8:
                    rel_plane = rel

                site_bond_params_list.append({
                    "site_pos": np.array(site_pos, dtype=float) + rel_plane,
                    "ind": idx + slab_atom_count,
                    "k": 0.35,
                    "deq": 0.0,
                })
        else:
            # Normal case: single adsorption atom to the site.
            for idx in ad_idx:
                site_bond_params_list.append({
                    "site_pos": site_pos,
                    "ind": idx + slab_atom_count,
                    "k": 0.5,
                    "deq": 0.0,
                })

        # Fallback for species without direct adsorption atom (e.g. H2):
        # restrain adsorbate center toward a point above the site to avoid free drift.
        if len(ad_idx) == 0:
            # center_bond_params_list.append({
            #     "indices": list(range(slab_atom_count, slab_atom_count + len(template["atoms"]))),
            #     "site_pos": site_pos,
            #     "k": 0.1,
            #     "deq": 0.0,
            # })

            # 选择吸附物的中心原子设置site_bond_params_list
            positions = template["atoms"].get_positions()
            geom_center = np.mean(positions, axis=0)
            center_local_idx = int(np.argmin(np.linalg.norm(positions - geom_center, axis=1)))
            site_bond_params_list.append({
                "site_pos": site_pos,
                "ind": center_local_idx + slab_atom_count,
                "k": 0.5,
                "deq": 0.0,
            })
            
        return site_bond_params_list, center_bond_params_list

    def _enumerate_azimuth_orientation(
        self,
        base_stru,
        ads_idx,
        myslab,
        pos,
        site_bond_params_list,
        center_bond_params_list,
        atom_bond_params_list=None,
    ):
        best_stru = base_stru
        best_angle = 0
        best_energy = None

        for az in self.rotation_angle_candidates:
            cand = base_stru.copy()
            if az != 0:
                geom.rotate_about_ads_vertical(cand, ads_idx, az, surface_normal=self.surface_normal)

            cand_shifted = cand.copy()
            cand_shifted.translate(np.array(pos, dtype=float) + self._site_lift_vector())
            cand_ads = myslab + cand_shifted
            cand_ads.set_constraint(self._build_constraints(cand_ads, surf_atom_num=len(myslab), with_internal_bonds=False))
            cand_ads.calc = constraint_utils.HarmonicallyForcedDP(
                model=MODEL,
                atom_bond_potentials=atom_bond_params_list,
                site_bond_potentials=site_bond_params_list,
                center_bond_potentials=center_bond_params_list,
            )

            try:
                e_cand = float(cand_ads.get_potential_energy())
            except Exception:
                continue

            if best_energy is None or e_cand < best_energy:
                best_energy = e_cand
                best_angle = az
                best_stru = cand

        return best_stru, best_angle, best_energy

    def _find_heavy_neighbor(self, stru, atom_idx, exclude_idx=None, cutoff_scale=1.25):
        """Find a likely bonded heavy-atom neighbor for torsion axis construction."""
        symbols = stru.get_chemical_symbols()
        pos = stru.get_positions()
        target_num = stru[atom_idx].number

        best_idx = None
        best_ratio = None
        for j in range(len(stru)):
            if j == atom_idx or j == exclude_idx:
                continue
            if symbols[j] == "H":
                continue

            d = np.linalg.norm(pos[j] - pos[atom_idx])
            r_sum = covalent_radii[target_num] + covalent_radii[stru[j].number]
            if r_sum <= 1e-12:
                continue

            ratio = d / r_sum
            if ratio <= cutoff_scale and (best_ratio is None or ratio < best_ratio):
                best_idx = j
                best_ratio = ratio

        return best_idx

    def _adsorbate_bond_set(self, atoms):
        """Return normalized intramolecular bond set for an adsorbate fragment."""
        bonds = set()
        for i, j in geom.get_bond_connections(atoms, shift=0, cutoff=1.2, bond_type="nosurf"):
            a, b = int(i), int(j)
            bonds.add((a, b) if a <= b else (b, a))
        return bonds

    def _is_adsorbate_structure_intact(self, xyz_path, sp_id):
        """Check whether adsorbate keeps template bond topology in adsorbate+slab structure."""
        if sp_id not in self.ads_templates:
            return False, "unknown_species"

        try:
            full_atoms = read(xyz_path)
        except Exception as exc:
            return False, f"read_failed: {exc}"

        slab_atom_count = len(self.slab.stru)
        template_atoms = self.ads_templates[sp_id]["atoms"]
        n_ads = len(template_atoms)

        if len(full_atoms) < slab_atom_count + n_ads:
            return False, "atom_count_mismatch"

        ads_atoms = full_atoms[slab_atom_count: slab_atom_count + n_ads]
        template_bonds = self._adsorbate_bond_set(template_atoms)
        current_bonds = self._adsorbate_bond_set(ads_atoms)

        if template_bonds != current_bonds:
            missing = sorted(template_bonds - current_bonds)
            extra = sorted(current_bonds - template_bonds)
            return False, f"bond_changed(missing={missing}, extra={extra})"

        return True, "ok"

    def _adsorbate_height_from_site(self, atoms, sp_id, site_pos):
        """Return projected adsorbate height along surface normal from adsorption site."""
        slab_atom_count = len(self.slab.stru)
        if len(atoms) <= slab_atom_count:
            return None

        normal = np.array(self.surface_normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            normal = np.array(self.surface_normal, dtype=float)
            normal_norm = np.linalg.norm(normal)
        normal /= normal_norm

        ads_positions = atoms.get_positions()[slab_atom_count:]
        rel = ads_positions - np.array(site_pos, dtype=float)
        proj = np.dot(rel, normal)
        return float(np.min(proj))

    def _is_adsorbate_close_to_surface(self, atoms, sp_id, site_pos, max_height=2.2):
        """Check whether adsorption center/anchor is sufficiently close to the surface site."""
        height = self._adsorbate_height_from_site(atoms, sp_id, site_pos)
        if height is None:
            return False, None
        return bool(height <= float(max_height)), float(height)

    def _get_rotating_group_indices(self, stru, moving_idx, pivot_idx):
        """Get indices connected to moving_idx without passing through pivot_idx."""
        n_atoms = len(stru)
        if n_atoms == 0:
            return []

        adjacency = [[] for _ in range(n_atoms)]
        for a, b in geom.get_bond_connections(stru, shift=0, cutoff=1.2, bond_type="nosurf"):
            adjacency[a].append(b)
            adjacency[b].append(a)

        visited = set([pivot_idx])
        stack = [moving_idx]
        group = []

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            for nb in adjacency[node]:
                if nb not in visited:
                    stack.append(nb)

        return group

    def _orient_broken_bond_torsion(
        self,
        stru,
        rec_bond,
        surface_normal,
        ads_indices=None,

    ):
        """Pre-orient broken-bond direction by rotating one endpoint-side fragment around a local axis."""
        if len(rec_bond) != 2:
            return False, None, 0.0

        i, j = rec_bond
        pos = stru.get_positions()

        surface_normal = self.surface_normal

        normal = np.array(surface_normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            return False, None, 0.0
        normal_unit = normal / normal_norm

        ads_set = set(ads_indices or [])
        bonded_pairs = self._adsorbate_bond_set(stru)

        def _passes_contact_constraints(cand_pos, moving_group):
            """Reject torsion candidates with unrealistically short non-bonded contacts."""
            n_atoms = len(stru)
            for a in moving_group:
                for b in range(n_atoms):
                    if a == b:
                        continue
                    pair = (a, b) if a <= b else (b, a)
                    if pair in bonded_pairs:
                        continue

                    d_ab = float(np.linalg.norm(cand_pos[a] - cand_pos[b]))
                    r_sum = float(covalent_radii[int(stru[a].number)] + covalent_radii[int(stru[b].number)])
                    d_min = max(0.75, 0.60 * r_sum)
                    if d_ab < d_min:
                        return False
            return True

        # Reference center of the adsorbate fragment, excluding the broken-bond pair when possible.
        ref_ids = [k for k in range(len(stru)) if k not in rec_bond]
        if not ref_ids:
            ref_ids = list(range(len(stru)))
        ref_center = np.mean(pos[ref_ids], axis=0)

        def _score(moved_point, pivot_idx):
            bond_vec = moved_point - pos[pivot_idx]
            bond_norm = np.linalg.norm(bond_vec)
            if bond_norm < 1e-12:
                return -1e9

            bond_unit = bond_vec / bond_norm
            # Prefer broken bond parallel to surface.
            parallel_score = 1.0 - abs(np.dot(bond_unit, normal_unit))

            # Prefer lateral (tilted) displacement from adsorbate center.
            rc_vec = moved_point - ref_center
            rc_norm = np.linalg.norm(rc_vec)
            if rc_norm < 1e-12:
                lateral_score = 0.0
            else:
                rc_plane = rc_vec - np.dot(rc_vec, normal_unit) * normal_unit
                lateral_score = np.linalg.norm(rc_plane) / rc_norm

            keep_ads_toward_surface_score = 0.0
            if ads_set:
                ad_center = np.mean(pos[list(ads_set)], axis=0)
                # Positive value means moved endpoint is more outward than adsorption atom(s).
                outward_delta = np.dot(moved_point - ad_center, normal_unit)
                keep_ads_toward_surface_score = np.tanh(outward_delta)

            return (
                0.65 * parallel_score
                + 0.20 * lateral_score
                + 0.15 * keep_ads_toward_surface_score
            )

        candidates = []
        for moving_idx, pivot_idx in [(i, j), (j, i)]:
            # Keep adsorption atoms fixed to preserve their orientation toward the surface.
            if moving_idx in ads_set:
                continue

            rotating_group = self._get_rotating_group_indices(stru, moving_idx, pivot_idx)
            if not rotating_group:
                continue
            # Avoid rotating the anchored adsorption atoms with the moving fragment.
            if any(idx in ads_set for idx in rotating_group):
                continue

            axis_end = self._find_heavy_neighbor(stru, pivot_idx, exclude_idx=moving_idx)
            if axis_end is None:
                continue

            axis_vec = pos[axis_end] - pos[pivot_idx]
            axis_norm = np.linalg.norm(axis_vec)
            if axis_norm < 1e-8:
                continue
            axis_unit = axis_vec / axis_norm

            base_point = pos[moving_idx].copy()
            for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                cand_point = geom.rotate_atom_around_axis(base_point, pos[pivot_idx], axis_unit, ang)

                cand_pos = pos.copy()
                for idx in rotating_group:
                    cand_pos[idx] = geom.rotate_atom_around_axis(
                        pos[idx].copy(), pos[pivot_idx], axis_unit, ang
                    )

                if not _passes_contact_constraints(cand_pos, rotating_group):
                    continue

                score = _score(cand_point, pivot_idx)
                candidates.append((score, moving_idx, pivot_idx, axis_unit, ang, rotating_group))

        if not candidates:
            return False, None, 0.0

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_moving_idx, best_pivot_idx, best_axis_unit, best_ang, best_group = candidates[0]

        if abs(best_ang) > 1e-8:
            for idx in best_group:
                pos[idx] = geom.rotate_atom_around_axis(
                    pos[idx].copy(), pos[best_pivot_idx], best_axis_unit, best_ang
                )
            stru.set_positions(pos)

        return True, int(best_moving_idx), float(best_ang)

    def _add_reactive_endpoint_site_anchor(
        self,
        stru,
        rec_bond,
        ads_indices,
        site_pos,
        site_bond_params_list,
        slab_atom_count,
        k=0.35,
        deq=0.6,
    ):
        """Add a weak site anchor for a likely flipping broken-bond endpoint during pre-TS relaxation."""
        if len(rec_bond) != 2:
            return False

        ads_set = set(ads_indices or [])
        candidates = [idx for idx in rec_bond if 0 <= idx < len(stru)]
        if not candidates:
            return False

        # Prefer the endpoint that is not adsorption atom and is farther outward along normal.
        normal = np.array(self.surface_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        pos = stru.get_positions()
        ad_center = np.mean(pos[list(ads_set)], axis=0) if ads_set else np.zeros(3)

        def _score(idx):
            penalty_ads = -100.0 if idx in ads_set else 0.0
            outward = float(np.dot(pos[idx] - ad_center, normal))
            return penalty_ads + outward

        anchor_local_idx = max(candidates, key=_score)
        anchor_global_idx = anchor_local_idx + slab_atom_count
        if any(p.get("ind") == anchor_global_idx for p in site_bond_params_list):
            return False

        site_bond_params_list.append({
            "site_pos": site_pos,
            "ind": anchor_global_idx,
            "k": k,
            "deq": deq,
        })
        return True

    def _get_imag_mode_report(self, ads, vib, rec_bond_global, slab_atom_count):
        """Analyze imaginary modes and compare bond-breaking vs adsorbate-drift character."""
        freqs = np.array(vib.get_frequencies(), dtype=complex)
        imag_indices = [
            i for i, f in enumerate(freqs)
            if (np.iscomplexobj(f) and abs(np.imag(f)) > 1e-12) or (np.isrealobj(f) and np.real(f) < 0.0)
        ]
        if len(imag_indices) == 0:
            return None

        normal = np.array(self.surface_normal, dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)

        n_atoms = len(ads)
        ads_indices = list(range(slab_atom_count, n_atoms))

        i, j = rec_bond_global
        pos = ads.get_positions()
        bond_vec = pos[j] - pos[i]
        bond_norm = np.linalg.norm(bond_vec)
        if bond_norm < 1e-12:
            bond_unit = np.array([1.0, 0.0, 0.0])
        else:
            bond_unit = bond_vec / bond_norm

        # Construct reaction-coordinate proxy vector for broken bond stretching.
        reaction_vec = np.zeros((n_atoms, 3), dtype=float)
        reaction_vec[i] = -bond_unit
        reaction_vec[j] = bond_unit
        reaction_norm = np.linalg.norm(reaction_vec)

        # Construct adsorbate translation along surface normal.
        ads_trans_vec = np.zeros((n_atoms, 3), dtype=float)
        for idx in ads_indices:
            ads_trans_vec[idx] = normal
        ads_trans_norm = np.linalg.norm(ads_trans_vec)

        mode_reports = []
        for mode_idx in imag_indices:
            mode = np.array(vib.get_mode(mode_idx), dtype=float)
            mode_norm = np.linalg.norm(mode)
            if mode_norm < 1e-14:
                continue

            atom_amp = np.linalg.norm(mode, axis=1)
            max_atom = int(np.argmax(atom_amp))

            bond_proj = 0.0
            drift_proj = 0.0
            if reaction_norm > 1e-14:
                bond_proj = abs(float(np.sum(mode * reaction_vec))) / (mode_norm * reaction_norm)
            if ads_trans_norm > 1e-14:
                drift_proj = abs(float(np.sum(mode * ads_trans_vec))) / (mode_norm * ads_trans_norm)

            mode_reports.append({
                "mode_idx": int(mode_idx),
                "freq": freqs[mode_idx],
                "bond_proj": float(bond_proj),
                "drift_proj": float(drift_proj),
                "max_atom": max_atom,
                "max_atom_symbol": ads[max_atom].symbol,
                "mode": mode,
            })

        if len(mode_reports) == 0:
            return None

        # Pick the most negative real component as primary imaginary mode.
        primary = min(mode_reports, key=lambda x: np.real(x["freq"]))
        return {
            "all": mode_reports,
            "primary": primary,
        }

    def _displace_and_relax_imag_mode(
        self,
        atoms,
        mode,
        sign,
        out_xyz,
        rec_bond_global,
        site_pos,
        fmax=0.10,
    ):
        """Displace structure along imaginary mode and perform short relaxation."""
        mode_norm = np.linalg.norm(mode)
        if mode_norm < 1e-14:
            return None

        disp_atoms = atoms.copy()
        scale = sign * self.imag_mode_displacement
        disp_atoms.set_positions(disp_atoms.get_positions() + scale * (mode / mode_norm))

        write(out_xyz.replace(".xyz", "_init.xyz"), disp_atoms)

        disp_atoms.calc = DP(model=MODEL)
        disp_atoms.set_constraint(self._build_constraints(disp_atoms, with_internal_bonds=False))

        try:
            opt = MDMin(disp_atoms, dt=0.05)
            opt.run(fmax=fmax, steps=self.imag_mode_relax_steps)
        except Exception:
            pass

        write(out_xyz, disp_atoms)

        i, j = rec_bond_global
        _, distances = get_distances(
            disp_atoms.get_positions(), None, disp_atoms.get_cell(), [True, True, True]
        )
        d_broken = float(distances[i, j])

        normal = np.array(self.surface_normal, dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)
        ads_positions = disp_atoms.get_positions()[len(self.slab.stru):]
        ads_center = np.mean(ads_positions, axis=0)
        height = float(np.dot(ads_center - np.array(site_pos, dtype=float), normal))

        return {
            "broken_bond": d_broken,
            "ads_height": height,
            "energy": float(disp_atoms.get_potential_energy()),
        }

    def _ts_bond_upper_bound(self, d_ref, atom_i_number, atom_j_number):
        """Build a stricter upper bound for TS reactive-bond length."""
        d_ref = float(d_ref)
        r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
        base_limit = max(
            self.ts_bond_max_scale_ref * d_ref,
            self.ts_bond_max_scale_covalent * r_sum,
        )
        additive_cap = d_ref + self.ts_bond_max_additive_cap
        return min(base_limit, additive_cap)

    def _is_ts_bond_overstretched(self, d_now, d_ref, atom_i_number, atom_j_number):
        """Return whether TS bond is too long and the applied upper bound."""
        d_max = self._ts_bond_upper_bound(d_ref, atom_i_number, atom_j_number)
        return bool(float(d_now) > float(d_max)), float(d_max)

    def _ts_bond_dissociation_threshold(self, d_ref, atom_i_number, atom_j_number):
        """Return a conservative lower bound indicating meaningful bond dissociation."""
        d_ref = float(d_ref)
        r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
        return min(
            self.ts_endpoint_dissoc_scale_ref * d_ref,
            self.ts_endpoint_dissoc_scale_covalent * r_sum,
            d_ref + self.ts_endpoint_dissoc_additive,
        )

    def _imag_frequency_magnitude(self, freq):
        """Return absolute imaginary magnitude for complex or negative-real frequencies."""
        if np.iscomplexobj(freq):
            return abs(float(np.imag(freq)))
        val = float(np.real(freq))
        if val < 0.0:
            return abs(val)
        return 0.0

    def _significant_imag_frequencies(self, freqs):
        """Keep only physically meaningful imaginary frequencies (exclude tiny numerical noise)."""
        cutoff = float(self.imag_freq_significant_cutoff)
        return [f for f in freqs if self._imag_frequency_magnitude(f) >= cutoff]



    def _get_vib_indices(self, atoms, slab_atom_count=None, surface_layer_tol=0.6):
        """Return ASE Vibrations indices for adsorbate atoms plus the first slab surface layer."""
        n_atoms = len(atoms)
        if n_atoms == 0:
            return []

        slab_total = len(self.slab.stru)
        if slab_atom_count is None:
            slab_atom_count = slab_total
        slab_atom_count = max(0, min(int(slab_atom_count), n_atoms))

        axis_id = {"x": 0, "y": 1, "z": 2}[self.normal_axis]
        positions = atoms.get_positions()

        if slab_atom_count <= 0:
            return list(range(n_atoms))

        slab_positions = positions[:slab_atom_count]
        top_coord = float(np.max(slab_positions[:, axis_id]))
        cutoff = top_coord - float(surface_layer_tol)
        surface_indices = [
            idx
            for idx in range(slab_atom_count)
            if float(slab_positions[idx, axis_id]) >= cutoff
        ]

        if n_atoms <= slab_atom_count:
            return surface_indices

        adsorbate_indices = list(range(slab_atom_count, n_atoms))
        return cfgmod._dedupe_keep_order(adsorbate_indices + surface_indices)


    def _thermo_analysis(self, atoms, T, name="vib", indices=None, delta=0.01, nfree=2):
        vib_mode_dir = f"{name}_mode"
        try:
            if indices is None:
                indices = self._get_vib_indices(atoms)
            name_abs = os.path.abspath(name)
            vib_mode_dir_abs = os.path.abspath(vib_mode_dir)
            os.makedirs(os.path.dirname(name_abs), exist_ok=True)

            cache_exists = False
            if os.path.isdir(name_abs):
                cache_entries = set(os.listdir(name_abs))
                cache_exists = (
                    os.path.exists(os.path.join(name_abs, "combined.json"))
                    or any(entry.startswith("cache.") and entry.endswith(".json") for entry in cache_entries)
                )

            if os.path.exists(vib_mode_dir_abs):
                shutil.rmtree(vib_mode_dir_abs)
            os.makedirs(vib_mode_dir_abs, exist_ok=True)

            vib = Vibrations(atoms, indices=indices, name=name_abs, delta=delta, nfree=nfree)
            if cache_exists:
                try:
                    vib.read()
                    print(f"Vibrations cache hit for {name_abs}; loaded existing results.")
                except Exception:
                    shutil.rmtree(name_abs)
                    vib.run()
            else:
                vib.run()
            vib.summary()

            vib_energies_raw = np.array(vib.get_energies(), dtype=complex)
            vib_energies = []
            for e in vib_energies_raw:
                if np.iscomplexobj(e) and abs(np.imag(e)) > 1e-10:
                    continue
                e_real = float(np.real(e))
                if e_real > 1e-12:
                    vib_energies.append(e_real)

            if len(vib_energies) == 0:
                return 0.0, 0.0, 0

            cwd = os.getcwd()
            os.chdir(vib_mode_dir_abs)
            vib.write_mode()
            os.chdir(cwd)

            thermo = HarmonicThermo(vib_energies, ignore_imag_modes=True)
            free_energy = thermo.get_helmholtz_energy(T)
            zpe = thermo.get_ZPE_correction()
            return float(free_energy), float(zpe), name_abs
        except Exception:
            return 0.0, 0.0, 0

    def _infer_gas_geometry(self, atoms):
        if len(atoms) == 1:
            return "monatomic"
        moments = np.array(atoms.get_moments_of_inertia(), dtype=float)
        if len(moments) == 0:
            return "nonlinear"
        if float(np.min(moments)) < 1e-3:
            return "linear"
        return "nonlinear"

    def _infer_gas_symmetry_number(self, atoms):
        formula = atoms.get_chemical_formula(mode="hill")
        if formula in {"H2", "N2", "O2", "F2", "Cl2", "Br2", "I2"}:
            return 2
        if formula in {"CO2", "H2O"}:
            return 2
        return 1

    def _gas_ideal_thermo_corrections(
        self,
        atoms,
        temperature_K,
        pressure_pa,
        name,
        indices=None,
        delta=0.01,
        nfree=2,
    ):
        if indices is None:
            indices = list(range(len(atoms)))

        name_abs = os.path.abspath(name)
        os.makedirs(os.path.dirname(name_abs), exist_ok=True)

        cache_exists = False
        if os.path.isdir(name_abs):
            cache_entries = set(os.listdir(name_abs))
            cache_exists = (
                os.path.exists(os.path.join(name_abs, "combined.json"))
                or any(entry.startswith("cache.") and entry.endswith(".json") for entry in cache_entries)
            )

        vib = Vibrations(atoms, indices=indices, name=name_abs, delta=delta, nfree=nfree)
        if cache_exists:
            try:
                vib.read()
                print(f"Vibrations cache hit for {name_abs}; loaded existing results.")
            except Exception:
                shutil.rmtree(name_abs)
                vib.run()
        else:
            vib.run()
        vib.summary()

        vib_energies_raw = np.array(vib.get_energies(), dtype=complex)
        vib_energies = []
        for e in vib_energies_raw:
            if np.iscomplexobj(e) and abs(np.imag(e)) > 1e-10:
                continue
            e_real = float(np.real(e))
            if e_real > 1e-12:
                vib_energies.append(e_real)

        if len(vib_energies) == 0:
            return 0.0, 0.0, 0

        geometry = self._infer_gas_geometry(atoms)
        symmetry_number = self._infer_gas_symmetry_number(atoms)
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            potentialenergy=0.0,
            atoms=atoms,
            geometry=geometry,
            symmetrynumber=symmetry_number,
            spin=0,
        )
        g_corr = thermo.get_gibbs_energy(temperature_K, pressure=pressure_pa)
        zpe = thermo.get_ZPE_correction()
        return float(g_corr), float(zpe), name_abs

    def generate_initial_adsorbate_guesses(self):
        slab = self.slab
        os.makedirs(os.path.join(self.path, "rxn", "Adsorbates"), exist_ok=True)

        self.valid_sites = {}
        self.valid_sites_all = {}

        def opt(
            path,
            prefix,
            site_bond_params_list,
            atom_bond_params_list=None,
            surf_atom_num=0,
            constraints=None,
            center_bond_params_list=None,
        ):
            struct = read(os.path.join(path, prefix + ".xyz"))
            if constraints is None:
                constraints = self._build_constraints(
                    struct,
                    surf_atom_num=surf_atom_num,
                    # with_internal_bonds=True,
                )
            struct.set_constraint(constraints)
            constraints = self._build_constraints(
                            struct,
                            surf_atom_num=surf_atom_num,
                            # with_internal_bonds=True,
                        )

            if not os.path.exists(os.path.join(path, prefix + "_opt.xyz")):
                calc = constraint_utils.HarmonicallyForcedDP(
                    model=MODEL,
                    atom_bond_potentials=atom_bond_params_list,
                    site_bond_potentials=site_bond_params_list,
                    center_bond_potentials=center_bond_params_list,
                )
                struct.calc = calc
                try:
                    opt_md = MDMin(struct, dt=0.05)
                    opt_md.run(fmax=0.5, steps=70)
                    write(os.path.join(path, prefix + "_weakopt.xyz"), struct)
                    calc = DP(model=MODEL)
                    struct.set_constraint(constraints)
                    struct.calc = calc
                    opt_qn = QuasiNewton(struct)
                    opt_qn.run(fmax=0.05, steps=70)
                except ase.calculators.calculator.CalculationFailed:
                    write(os.path.join(path, prefix + "_opt_fail.xyz"), struct)
                    return None
                write(os.path.join(path, prefix + "_opt.xyz"), struct)
            else:
                struct = read(os.path.join(path, prefix + "_opt.xyz"))
                struct.set_constraint(constraints)
                struct.calc = DP(model=MODEL)

            try:
                return float(struct.get_potential_energy())
            except Exception:
                return None

        for sp_id, data in self.ads_templates.items():
            ads_dir = self._adsorbate_dir_candidates(sp_id)[0]
            os.makedirs(ads_dir, exist_ok=True)
            sites = slab.unique_sites["idx"]
            myslab = slab.stru.copy()
            energy = {}

            for site in sites:
                path = ads_dir
                prefix = f"{site}"
                tagged_prefix = self._tagged_stem(prefix)
                pos = slab.get_site(site)
                stru = data["atoms"].copy()

                site_bond_params_list, center_bond_params_list = self._build_site_and_center_anchors(
                    data, pos, len(myslab)
                )
                # for p in site_bond_params_list:
                #     p["k"] = 0.3
                # for p in center_bond_params_list:
                #     p["k"] = 0.3

                if self.enable_rotation_enum_ads:
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

                stru.translate(np.array(pos, dtype=float) + self._site_lift_vector())
                ads = myslab + stru
                write(os.path.join(path, f"{tagged_prefix}.xyz"), ads)
                e_site = opt(
                    path,
                    tagged_prefix,
                    site_bond_params_list,
                    [],
                    len(myslab),
                    center_bond_params_list=center_bond_params_list,
                )
                if e_site is not None:
                    opt_xyz = os.path.join(path, f"{tagged_prefix}_opt.xyz")
                    if os.path.exists(opt_xyz):
                        intact, reason = self._is_adsorbate_structure_intact(opt_xyz, sp_id)
                        if not intact:
                            print(
                                f"Reject site {site} for {sp_id}: "
                                f"dissociated adsorbate after opt ({reason})"
                            )
                            e_site = None
                        else:
                            opt_atoms = read(opt_xyz)
                            close_ok, height = self._is_adsorbate_close_to_surface(
                                opt_atoms,
                                sp_id,
                                pos,
                                max_height=2,
                            )
                            if not close_ok:
                                print(
                                    f"Site {site} for {sp_id} is too far from surface "
                                    f"(height={height:.3f} A). Run rescue optimization."
                                )

                                rescue_atoms = opt_atoms.copy()
                                normal = np.array(self.surface_normal, dtype=float)
                                normal_norm = np.linalg.norm(normal)
                                if normal_norm < 1e-12:
                                    normal = np.array(self.surface_normal, dtype=float)
                                    normal_norm = np.linalg.norm(normal)
                                normal /= normal_norm

                                target_height = 1.3
                                pull_dist = max(0.0, float(height) - target_height)
                                if pull_dist > 1e-8:
                                    rescue_pos = rescue_atoms.get_positions()
                                    rescue_pos[len(myslab):] -= pull_dist * normal
                                    rescue_atoms.set_positions(rescue_pos)

                                rescue_site_params, rescue_center_params = self._build_site_and_center_anchors(
                                    data,
                                    pos,
                                    len(myslab),
                                )
                                for p in rescue_site_params:
                                    p["k"] = max(float(p.get("k", 0.5)), 1.2)
                                for p in rescue_center_params:
                                    p["k"] = max(float(p.get("k", 0.5)), 0.8)

                                rescue_atoms.set_constraint(
                                    self._build_constraints(rescue_atoms, surf_atom_num=len(myslab), with_internal_bonds=False)
                                )
                                rescue_atoms.calc = constraint_utils.HarmonicallyForcedDP(
                                    model=MODEL,
                                    atom_bond_potentials=[],
                                    site_bond_potentials=rescue_site_params,
                                    center_bond_potentials=rescue_center_params,
                                )
                                try:
                                    rescue_md = MDMin(rescue_atoms, dt=0.05)
                                    rescue_md.run(fmax=0.4, steps=80)
                                    rescue_qn = QuasiNewton(rescue_atoms)
                                    rescue_qn.run(fmax=0.08, steps=80)
                                except ase.calculators.calculator.CalculationFailed:
                                    pass

                                # Final unbiased relaxation and energy evaluation.
                                rescue_atoms.calc = DP(model=MODEL)
                                rescue_atoms.set_constraint(
                                    self._build_constraints(rescue_atoms, surf_atom_num=len(myslab), with_internal_bonds=False)
                                )
                                try:
                                    release_qn = QuasiNewton(rescue_atoms)
                                    release_qn.run(fmax=0.05, steps=60)
                                except ase.calculators.calculator.CalculationFailed:
                                    pass

                                write(opt_xyz, rescue_atoms)
                                try:
                                    e_site = float(rescue_atoms.get_potential_energy())
                                except Exception:
                                    e_site = None

                                if e_site is not None:
                                    intact2, reason2 = self._is_adsorbate_structure_intact(opt_xyz, sp_id)
                                    close_ok2, height2 = self._is_adsorbate_close_to_surface(
                                        rescue_atoms,
                                        sp_id,
                                        pos,
                                        max_height=2.2,
                                    )
                                    if (not intact2) or (not close_ok2):
                                        print(
                                            f"Reject site {site} for {sp_id} after rescue: "
                                            f"intact={intact2}, height={height2}, reason={reason2}"
                                        )
                                        e_site = None
                energy[site] = e_site

            valid_energy = {k: v for k, v in energy.items() if v is not None}
            if valid_energy:
                valid_sorted = dict(sorted(valid_energy.items(), key=lambda item: item[1]))
                valid_sites = list(valid_sorted.keys())
            else:
                print(
                    f"No intact adsorption structures found for {sp_id}; "
                    "skip bondfix retry and force-select one fallback site."
                )
                fallback_energy = {}
                for site in sites:
                    tagged_prefix = self._tagged_stem(f"{site}")
                    opt_xyz = os.path.join(ads_dir, f"{tagged_prefix}_opt.xyz")
                    if not os.path.exists(opt_xyz):
                        continue
                    try:
                        atoms = read(opt_xyz)
                        atoms.set_constraint(
                            self._build_constraints(
                                atoms,
                                surf_atom_num=len(myslab),
                                with_internal_bonds=False,
                            )
                        )
                        atoms.calc = DP(model=MODEL)
                        fallback_energy[site] = float(atoms.get_potential_energy())
                    except Exception:
                        continue

                if fallback_energy:
                    chosen_site = min(fallback_energy, key=fallback_energy.get)
                    chosen_energy = float(fallback_energy[chosen_site])
                    valid_sites = [chosen_site]
                    energy[chosen_site] = chosen_energy
                    print(
                        f"Fallback for {sp_id}: select site {chosen_site} "
                        f"(E={chosen_energy:.6f} eV) without bondfix retry."
                    )
                else:
                    if len(sites) > 0:
                        chosen_site = sites[0]
                        valid_sites = [chosen_site]
                        print(
                            f"Fallback for {sp_id}: no readable *_opt.xyz; "
                            f"force-select first site {chosen_site}."
                        )
                    else:
                        valid_sites = []
                        print(f"Fallback for {sp_id}: no available site to select.")

            valid_sites_all = list(valid_sites)
            if self.top_x is not None and self.top_x > 0:
                valid_sites_preferred = valid_sites_all[:self.top_x]
            else:
                valid_sites_preferred = valid_sites_all

            self.valid_sites_all[sp_id] = valid_sites_all
            self.valid_sites[sp_id] = valid_sites_preferred
            print(
                f"Valid sites for {sp_id}: preferred={valid_sites_preferred}, "
                f"all={valid_sites_all}"
            )

        for rxn in self.rxns_dict:
            rxn["valid_reactant_sites"] = []
            rxn["valid_reactant_sites_all"] = []
            rxn["valid_product_sites"] = []
            for sp_id in rxn["reactant_species"]:
                rxn["valid_reactant_sites"] = self.valid_sites.get(sp_id, slab.unique_sites["idx"])
                rxn["valid_reactant_sites_all"] = self.valid_sites_all.get(sp_id, slab.unique_sites["idx"])
            for sp_id in rxn["product_species"]:
                rxn["valid_product_sites"] = self.valid_sites.get(sp_id, slab.unique_sites["idx"])

    def generate_rxn_ts_guesses_ccqn(self):
        slab = self.slab
        os.makedirs(os.path.join(self.path, "rxn", "TS_guesses"), exist_ok=True)

        # Reuse previously archived TS records so reruns do not overwrite them with empty content.
        old_records = self._load_ts_records()
        records_by_tag = {rec.get("tag"): rec for rec in old_records if isinstance(rec, dict) and rec.get("tag")}
        self.ts_records = list(records_by_tag.values())

        def _drop_record_by_tag(tag):
            self.ts_records = [rec for rec in self.ts_records if rec.get("tag") != tag]

        def _bond_key(i, j):
            a, b = int(i), int(j)
            return (a, b) if a <= b else (b, a)

        for idx, rxn in enumerate(self.rxns_dict):
            preferred_sites = list(rxn.get("valid_reactant_sites", []))
            all_sites = list(rxn.get("valid_reactant_sites_all", preferred_sites))
            sites = cfgmod._dedupe_keep_order(preferred_sites + [s for s in all_sites if s not in preferred_sites])
            backup_sites = set(s for s in sites if s not in preferred_sites)
            accepted_any_site = False
            rec_bond = rxn["broken_bond"]
            sp_id = rxn["reactant_species"][0]
            rxn_key = self.rxn_key_by_index[idx]
            myslab = slab.stru.copy()
            template = self.ads_templates[sp_id]
            
            for site in sites:
                if site in backup_sites and accepted_any_site:
                    continue
                try:
                    ts_stem = self._tagged_stem(f"{rxn_key}_site_{site}")
                    legacy_ts_stem = self._tagged_stem(f"rxn_{idx}_site_{site}")
                    ts_tag = ts_stem
                    ts_guess_xyz = os.path.join(self.path, "rxn", "TS_guesses", f"{ts_stem}.xyz")
                    legacy_ts_guess_xyz = os.path.join(self.path, "rxn", "TS_guesses", f"{legacy_ts_stem}.xyz")
                    ts_opt_xyz = os.path.join(self.path, "rxn", "TS_guesses", f"{ts_stem}_opt.xyz")
                    legacy_ts_opt_xyz = os.path.join(self.path, "rxn", "TS_guesses", f"{legacy_ts_stem}_opt.xyz")
                    out_xyz = os.path.join(self.path, "rxn", "TS_guesses", f"{ts_stem}_opt_ccqn.xyz")
                    legacy_out_xyz = os.path.join(self.path, "rxn", "TS_guesses", f"{legacy_ts_stem}_opt_ccqn.xyz")
                    summary_log = self._tagged_stem(f"optimization_summary_{rxn_key}_site_{site}") + ".log"
                    legacy_summary_log = self._tagged_stem(f"optimization_summary_{idx}_site_{site}") + ".log"

                    reuse_out_xyz = out_xyz
                    if (not os.path.exists(reuse_out_xyz)) and os.path.exists(legacy_out_xyz):
                        reuse_out_xyz = legacy_out_xyz

                    reuse_summary_log = summary_log if os.path.exists(summary_log) else legacy_summary_log
                    if not os.path.exists(reuse_summary_log):
                        reuse_summary_log = summary_log

                    # Prefer the optimized pre-TS structure as IRC initial state.
                    reuse_initial_xyz = ts_opt_xyz
                    if not os.path.exists(reuse_initial_xyz) and os.path.exists(legacy_ts_opt_xyz):
                        reuse_initial_xyz = legacy_ts_opt_xyz
                    if not os.path.exists(reuse_initial_xyz) and os.path.exists(ts_guess_xyz):
                        reuse_initial_xyz = ts_guess_xyz
                    if not os.path.exists(reuse_initial_xyz) and os.path.exists(legacy_ts_guess_xyz):
                        reuse_initial_xyz = legacy_ts_guess_xyz
                    if not os.path.exists(reuse_initial_xyz):
                        reuse_initial_xyz = reuse_out_xyz

                    # If TS result already exists, only read/reuse and skip expensive optimization.
                    if os.path.exists(reuse_out_xyz):
                        print(f"Reuse existing TS result for {ts_stem}: {reuse_out_xyz}")

                        if not os.path.exists(reuse_summary_log):
                            with open(reuse_summary_log, "w") as f:
                                f.write(
                                    "Skipped generate_rxn_ts_guesses_ccqn recomputation: "
                                    "existing _opt_ccqn structure was reused.\n"
                                )

                        if ts_tag not in records_by_tag:
                            self._save_ts_record(
                                rxn_idx=idx,
                                site=site,
                                sp_id=sp_id,
                                rxn_key=rxn_key,
                                rec_bond=rec_bond,
                                slab_atom_count=len(myslab),
                                initial_xyz=reuse_initial_xyz,
                                ts_xyz=reuse_out_xyz,
                                summary_log=reuse_summary_log,
                                tag=ts_tag,
                            )
                            records_by_tag[ts_tag] = self.ts_records[-1]
                        continue

                    pos = slab.get_site(site)
                    print(f"Processing rxn_{idx} at site {site} with rec_bond {rec_bond} and sp_id {sp_id}")

                    stru_seed = template["atoms"].copy()
                    template_bonds = {
                        _bond_key(i, j)
                        for i, j in geom.get_bond_connections(stru_seed, shift=0, cutoff=1.2, bond_type="nosurf")
                    }
                    expected_broken_bond = _bond_key(rec_bond[0], rec_bond[1])

                    _, distances = get_distances(
                        stru_seed.get_positions(), None, stru_seed.get_cell(), [True, True, True]
                    )
                    d = distances[rec_bond[0], rec_bond[1]]
                    msg_header = f"distance between broken bond atoms before adjustment: {d:.2f} Å\n"

                    atom_bond_params_list = [{
                        "ind1": rec_bond[0] + len(myslab),
                        "ind2": rec_bond[1] + len(myslab),
                        "k": 2.0,
                        "deq": 1.1 * d,
                    }]

                    torsion_info = self._orient_broken_bond_torsion(
                        stru_seed,
                        rec_bond=rec_bond,
                        ads_indices=template["ad_idx"],
                        surface_normal=self.surface_normal,
                    )

                    msg_seed = msg_header
                    if torsion_info:
                        torsion_ok, torsion_atom_idx, torsion_angle = torsion_info
                        if torsion_ok:
                            msg_seed += (
                                f"pre-rotated broken-bond torsion: atom {torsion_atom_idx} "
                                f"by {torsion_angle:.1f} deg\n"
                            )

                    # 如果断键与表面法线夹角过小，绕吸附原子按内角差值旋转，避免断键过于垂直表面
                    angle_before, angle_rot = geom.rotate_about_ads_inner_angle(
                        stru_seed,
                        rec_bond=rec_bond,
                        ads_indices=template["ad_idx"],
                        target_min_angle_deg=50.0,
                        surface_normal=self.surface_normal,
                    )
                    if angle_rot > 0.0:
                        msg_seed += (
                            f"rotated around ads atom by inner-angle rule: "
                            f"{angle_before:.2f}° -> >= 50.00°, applied {angle_rot:.2f}°\n"
                        )

                    azimuth_candidates = [0.0]
                    if self.enable_rotation_enum_ts:
                        site_bond_params_tmp, center_bond_params_tmp = self._build_site_and_center_anchors(
                            template, pos, len(myslab)
                        )
                        # for p in site_bond_params_tmp:
                        #     p["k"] = 0.5
                        # for p in center_bond_params_tmp:
                        #     p["k"] = 0.3
                        _, best_angle, best_energy = self._enumerate_azimuth_orientation(
                            base_stru=stru_seed.copy(),
                            ads_idx=template["ad_idx"],
                            myslab=myslab,
                            pos=pos,
                            site_bond_params_list=site_bond_params_tmp,
                            center_bond_params_list=center_bond_params_tmp,
                            atom_bond_params_list=atom_bond_params_list,
                        )
                        azimuth_candidates = [float(best_angle)] + [
                            float(a) for a in self.rotation_angle_candidates if float(a) != float(best_angle)
                        ]
                        if best_energy is not None:
                            msg_seed += (
                                f"azimuth enumeration selected initial angle {best_angle:.1f}° "
                                f"with E={best_energy:.6f} eV\n"
                            )
                        else:
                            msg_seed += "azimuth enumeration failed; start from default angle 0°\n"

                    accepted = False
                    attempt_messages = []

                    for attempt_idx, az_angle in enumerate(azimuth_candidates, start=1):
                        attempt_msg = msg_seed
                        attempt_msg += (
                            f"Attempt {attempt_idx}/{len(azimuth_candidates)}: "
                            f"azimuth={az_angle:.1f}°\n"
                        )

                        site_bond_params_list, center_bond_params_list = self._build_site_and_center_anchors(
                            template, pos, len(myslab)
                        )
                        # for p in site_bond_params_list:
                        #     p["k"] = 0.5
                        # for p in center_bond_params_list:
                        #     p["k"] = 0.3

                        stru = stru_seed.copy()
                        if abs(float(az_angle)) > 1e-8:
                            geom.rotate_about_ads_vertical(
                                stru=stru,
                                ads_idx=template["ad_idx"],
                                angle_deg=float(az_angle),
                                surface_normal=self.surface_normal,
                            )

                        stru.translate(np.array(pos, dtype=float) + self._site_lift_vector())
                        ads = myslab + stru
                        write(ts_guess_xyz, ads)

                        endpoint_anchor_added = self._add_reactive_endpoint_site_anchor(
                            stru=stru,
                            rec_bond=rec_bond,
                            ads_indices=template["ad_idx"],
                            site_pos=pos,
                            site_bond_params_list=site_bond_params_list,
                            slab_atom_count=len(myslab),
                        )
                        if endpoint_anchor_added:
                            attempt_msg += "added weak endpoint-site anchor during pre-TS optimization\n"

                        calc = constraint_utils.HarmonicallyForcedDP(
                            model=MODEL,
                            atom_bond_potentials=atom_bond_params_list,
                            site_bond_potentials=site_bond_params_list,
                            center_bond_potentials=center_bond_params_list,
                        )
                        ads.calc = calc
                        ads.set_constraint(self._build_constraints(ads, with_internal_bonds=False))

                        try:
                            opt_md = MDMin(ads, dt=0.05)
                            opt_md.run(fmax=0.5, steps=70)
                            opt_qn = QuasiNewton(ads)
                            opt_qn.run(fmax=0.05, steps=70)
                        except ase.calculators.calculator.CalculationFailed as exc:
                            attempt_msg += f"Optimization failed: {exc}\n"
                            attempt_messages.append(attempt_msg)
                            continue

                        ads.calc = DP(model=MODEL)
                        _, distances = get_distances(
                            ads.get_positions(), None, stru.get_cell(), [True, True, True]
                        )
                        rec_bond_global = (rec_bond[0] + len(myslab), rec_bond[1] + len(myslab))
                        d_opt = distances[rec_bond_global[0], rec_bond_global[1]]
                        attempt_msg += f"distance between broken bond atoms after opt: {d_opt:.2f} Å\n"

                        z_i = int(ads[rec_bond_global[0]].number)
                        z_j = int(ads[rec_bond_global[1]].number)
                        d_ccqn_max = self._ts_bond_upper_bound(d_ref=d, atom_i_number=z_i, atom_j_number=z_j)
                        attempt_msg += f"TS reactive-bond max length threshold: {d_ccqn_max:.2f} Å\n"

                        write(os.path.join(self.path, "rxn", "TS_guesses", f"{ts_stem}_opt.xyz"), ads)
                        ads_pre_ccqn = ads.copy()

                        ads.calc = DP(model=MODEL)
                        opt_ccqn = CCQN(
                            ads,
                            logfile=os.path.join(
                                self.path,
                                "rxn",
                                "TS_guesses",
                                f"{ts_stem}_ccqn_try{attempt_idx}.log",
                            ),
                            e_vector_method="ic",
                            reactive_bonds=[(rec_bond[0] + len(myslab), rec_bond[1] + len(myslab))],
                            saddle_optimizer="sella",
                            sella_kwargs={"delta0": 0.05},
                        )
                        opt_ccqn.run(fmax=0.05, steps=100)
                        write(out_xyz, ads)

                        _, distances = get_distances(
                            ads.get_positions(), None, stru.get_cell(), [True, True, True]
                        )
                        d_ccqn = distances[rec_bond_global[0], rec_bond_global[1]]
                        attempt_msg += f"distance between broken bond atoms after ccqn: {d_ccqn:.2f} Å\n"

                        overstretched, _ = self._is_ts_bond_overstretched(
                            d_now=d_ccqn,
                            d_ref=d,
                            atom_i_number=z_i,
                            atom_j_number=z_j,
                        )
                        if overstretched:
                            attempt_msg += (
                                "Reactive bond is over-elongated after CCQN; "
                                "rollback to pre-CCQN structure and rerun conservative CCQN.\n"
                            )
                            ads = ads_pre_ccqn.copy()
                            ads.calc = DP(model=MODEL)
                            ads.set_constraint(self._build_constraints(ads, with_internal_bonds=False))
                            opt_ccqn_retry = CCQN(
                                ads,
                                logfile=os.path.join(
                                    self.path,
                                    "rxn",
                                    "TS_guesses",
                                    f"{ts_stem}_ccqn_retry_try{attempt_idx}.log",
                                ),
                                e_vector_method="ic",
                                reactive_bonds=[rec_bond_global],
                                saddle_optimizer="sella",
                                sella_kwargs={"delta0": 0.02},
                            )
                            opt_ccqn_retry.run(fmax=0.04, steps=70)
                            write(out_xyz, ads)

                            _, distances = get_distances(
                                ads.get_positions(), None, stru.get_cell(), [True, True, True]
                            )
                            d_ccqn_retry = distances[rec_bond_global[0], rec_bond_global[1]]
                            attempt_msg += (
                                "distance between broken bond atoms after rollback ccqn: "
                                f"{d_ccqn_retry:.2f} Å\n"
                            )

                            overstretched_retry, _ = self._is_ts_bond_overstretched(
                                d_now=d_ccqn_retry,
                                d_ref=d,
                                atom_i_number=z_i,
                                atom_j_number=z_j,
                            )
                            if overstretched_retry:
                                attempt_msg += (
                                    "Rejected this orientation: reactive bond is still over-elongated "
                                    "after rollback retry.\n"
                                )
                                attempt_messages.append(attempt_msg)
                                continue

                            d_ccqn = d_ccqn_retry
                            attempt_msg += "Rollback retry recovered acceptable TS bond length.\n"

                        # Check TS whether dissociation
                        d_ccqn_min = self._ts_bond_dissociation_threshold(
                            d_ref=d,
                            atom_i_number=z_i,
                            atom_j_number=z_j,
                        )
                        attempt_msg += (
                            f"TS reactive-bond dissociation min threshold: {d_ccqn_min:.2f} Å\n"
                        )
                        if d_ccqn < d_ccqn_min:
                            attempt_msg += (
                                "Rejected this orientation: reactive bond is not sufficiently "
                                "dissociated after CCQN; skip vibration checks.\n"
                            )
                            attempt_messages.append(attempt_msg)
                            continue

                        ads.calc = DP(model=MODEL)
                        vib_prefix = os.path.join(
                            self.path,
                            "rxn",
                            "TS_guesses",
                            f"{ts_stem}_vib_try{attempt_idx}",
                        )

                        def _run_vibration_with_cleanup(prefix, indices=None):
                            if os.path.exists(prefix):
                                shutil.rmtree(prefix)
                            if indices is None:
                                indices = self._get_vib_indices(ads)
                            vib_local = Vibrations(ads, name=prefix, indices=indices)
                            vib_local.run()
                            vib_local.summary()
                            freqs_local = vib_local.get_frequencies()
                            imag_local = [
                                f
                                for f in freqs_local
                                if (np.iscomplexobj(f) and abs(np.imag(f)) > 1e-12)
                                or (np.isrealobj(f) and np.real(f) < 0.0)
                            ]
                            return vib_local, freqs_local, imag_local

                        vib, freqs, imag_freqs = _run_vibration_with_cleanup(vib_prefix)
                        attempt_msg += f"Initial imaginary frequency count: {len(imag_freqs)}\n"

                        if len(imag_freqs) != 1:
                            attempt_msg += (
                                "Imaginary frequency count is not 1; run an extra Sella saddle optimization "
                                "and recompute vibrations.\n"
                            )
                            try:
                                ads.calc = DP(model=MODEL)
                                ads.set_constraint(self._build_constraints(ads, with_internal_bonds=False))
                                retry_traj = os.path.join(
                                    self.path,
                                    "rxn",
                                    "TS_guesses",
                                    f"{ts_stem}_sella_retry_try{attempt_idx}.traj",
                                )
                                retry_log = os.path.join(
                                    self.path,
                                    "rxn",
                                    "TS_guesses",
                                    f"{ts_stem}_sella_retry_try{attempt_idx}.log",
                                )
                                opt_retry = Sella(ads, trajectory=retry_traj, logfile=retry_log)
                                opt_retry.run(fmax=0.05, steps=80)

                                retry_xyz = os.path.join(
                                    self.path,
                                    "rxn",
                                    "TS_guesses",
                                    f"{ts_stem}_opt_sella_retry_try{attempt_idx}.xyz",
                                )
                                write(retry_xyz, ads)

                                _, distances = get_distances(
                                    ads.get_positions(), None, stru.get_cell(), [True, True, True]
                                )
                                d_retry = distances[rec_bond[0] + len(myslab), rec_bond[1] + len(myslab)]
                                attempt_msg += (
                                    f"distance between broken bond atoms after extra sella: {d_retry:.2f} Å\n"
                                )

                                vib, freqs, imag_freqs = _run_vibration_with_cleanup(vib_prefix)
                                attempt_msg += (
                                    f"Imaginary frequency count after extra sella: {len(imag_freqs)}\n"
                                )
                            except Exception as exc:
                                attempt_msg += f"Extra sella saddle optimization failed: {exc}\n"

                        significant_imag_freqs = self._significant_imag_frequencies(freqs)
                        attempt_msg += (
                            "Significant imaginary frequency count "
                            f"(|nu| >= {self.imag_freq_significant_cutoff:.1f} cm^-1): "
                            f"{len(significant_imag_freqs)}\n"
                        )
                        if len(significant_imag_freqs) != 1:
                            attempt_msg += (
                                "Rejected this orientation: significant imaginary frequency "
                                "count is not 1 after TS refinement.\n"
                            )
                            attempt_messages.append(attempt_msg)
                            continue

                        ads_only = ads[len(myslab):]
                        final_bonds = {
                            _bond_key(i, j)
                            for i, j in geom.get_bond_connections(
                                ads_only,
                                shift=0,
                                cutoff=1.2,
                                bond_type="nosurf",
                            )
                        }
                        broken_bonds = sorted(template_bonds - final_bonds)
                        unexpected_broken_bonds = [b for b in broken_bonds if b != expected_broken_bond]
                        has_unexpected_break = len(unexpected_broken_bonds) > 0

                        if has_unexpected_break:
                            attempt_msg += (
                                f"Unexpected broken bonds detected after CCQN: "
                                f"{unexpected_broken_bonds}\n"
                            )

                        if len(imag_freqs) != 1 and has_unexpected_break:
                            attempt_msg += (
                                "Rejected this orientation: non-single imaginary frequency "
                                "and unexpected bond breaking were both observed.\n"
                            )
                            attempt_messages.append(attempt_msg)
                            continue

                        if imag_freqs:
                            attempt_msg += f"Imaginary frequencies found: {imag_freqs}\n"
                        else:
                            attempt_msg += "No imaginary frequencies found (not a TS candidate).\n"

                        imag_report = self._get_imag_mode_report(
                            ads=ads,
                            vib=vib,
                            rec_bond_global=rec_bond_global,
                            slab_atom_count=len(myslab),
                        )
                        if imag_report is not None:
                            for m in imag_report["all"]:
                                attempt_msg += (
                                    "Imag mode "
                                    f"{m['mode_idx']}: f={m['freq']}, "
                                    f"bond_proj={m['bond_proj']:.3f}, "
                                    f"ads_drift_proj={m['drift_proj']:.3f}, "
                                    f"max_atom={m['max_atom_symbol']}{m['max_atom']}\n"
                                )

                            primary = imag_report["primary"]
                            if primary["bond_proj"] >= primary["drift_proj"]:
                                attempt_msg += "Primary imaginary mode is bond-breaking dominated.\n"
                            else:
                                attempt_msg += "Warning: primary imaginary mode is adsorbate-drift dominated.\n"

                            drift_dominated = (
                                primary["bond_proj"] + self.ts_primary_mode_bond_margin
                                < primary["drift_proj"]
                            )
                            weak_bond_character = (
                                primary["bond_proj"] < self.ts_primary_mode_min_bond_proj
                            )
                            if drift_dominated or weak_bond_character:
                                attempt_msg += (
                                    "Rejected this orientation: primary imaginary mode does not "
                                    "show sufficiently strong bond-breaking character.\n"
                                )
                                attempt_messages.append(attempt_msg)
                                continue

                            if self.enable_imag_mode_check:
                                plus_xyz = os.path.join(
                                    self.path,
                                    "rxn",
                                    "TS_guesses",
                                    f"{ts_stem}_imag_plus_try{attempt_idx}.xyz",
                                )
                                minus_xyz = os.path.join(
                                    self.path,
                                    "rxn",
                                    "TS_guesses",
                                    f"{ts_stem}_imag_minus_try{attempt_idx}.xyz",
                                )
                                plus_info = self._displace_and_relax_imag_mode(
                                    atoms=ads,
                                    mode=primary["mode"],
                                    sign=+1.0,
                                    out_xyz=plus_xyz,
                                    rec_bond_global=rec_bond_global,
                                    site_pos=pos,
                                )
                                minus_info = self._displace_and_relax_imag_mode(
                                    atoms=ads,
                                    mode=primary["mode"],
                                    sign=-1.0,
                                    out_xyz=minus_xyz,
                                    rec_bond_global=rec_bond_global,
                                    site_pos=pos,
                                )
                                if plus_info is not None and minus_info is not None:
                                    attempt_msg += (
                                        f"Imag+ endpoint: E={plus_info['energy']:.6f} eV, "
                                        f"d_break={plus_info['broken_bond']:.3f} A, "
                                        f"h_ads={plus_info['ads_height']:.3f} A\n"
                                    )
                                    attempt_msg += (
                                        f"Imag- endpoint: E={minus_info['energy']:.6f} eV, "
                                        f"d_break={minus_info['broken_bond']:.3f} A, "
                                        f"h_ads={minus_info['ads_height']:.3f} A\n"
                                    )
                        else:
                            attempt_msg += "Imaginary mode vectors unavailable for projection analysis.\n"

                        attempt_messages.append(attempt_msg)
                        accepted = True
                        break

                    msg = "\n".join(attempt_messages) + "\n"
                    with open(summary_log, "w") as f:
                        f.write(msg)

                    if not accepted:
                        print(
                            f"All orientations rejected for rxn_{idx}_site_{site}; "
                            "try next candidate site."
                        )
                        continue

                    accepted_any_site = True

                    if ts_tag in records_by_tag:
                        _drop_record_by_tag(ts_tag)
                        records_by_tag.pop(ts_tag, None)

                    self._save_ts_record(
                        rxn_idx=idx,
                        site=site,
                        sp_id=sp_id,
                        rxn_key=rxn_key,
                        rec_bond=rec_bond,
                        slab_atom_count=len(myslab),
                        initial_xyz=ts_opt_xyz if os.path.exists(ts_opt_xyz) else ts_guess_xyz,
                        ts_xyz=out_xyz,
                        summary_log=summary_log,
                        tag=ts_tag,
                    )
                    records_by_tag[ts_tag] = self.ts_records[-1]
                except Exception as exc:
                    print(f"Error for rxn_{idx}_site_{site}, skip this structure: {exc}")
                    traceback.print_exc()
                    continue

            if (not accepted_any_site) and len(backup_sites) > 0:
                print(
                    f"No TS accepted on preferred top-x sites for rxn_{idx}; "
                    "backup sites were also attempted but all failed."
                )

        self._flush_ts_records()

    def get_final_state_IRC(self):
        try:
            from sella import IRC
        except Exception as exc:
            raise RuntimeError(f"Sella IRC is unavailable: {exc}")

        records = self._load_ts_records()
        if len(records) == 0:
            print("No TS records found; skip IRC final-state search.")
            return []

        out_root = os.path.join(self.path, "rxn", "IRC_final_states")
        os.makedirs(out_root, exist_ok=True)

        temperature_K = float(self.temperature)
        enable_irc_thermo = bool(self.enable_irc_thermo_corrections)
        vib_corr_cache = {}

        if enable_irc_thermo:
            vib_root = os.path.join(out_root, "vib_jobs")
            os.makedirs(vib_root, exist_ok=True)
            print(f"IRC thermo corrections enabled at T={temperature_K:.2f} K")
        else:
            print("IRC thermo corrections disabled; all ZPE/G correction terms are set to 0.")

        def _vib_corrections_for_atoms(atoms, cache_key, tag_prefix):
            if not enable_irc_thermo:
                return {
                    "zpe": 0.0,
                    "g_corr": 0.0,
                    "used_prefix": 0,
                    "has_vib": False,
                }

            if cache_key in vib_corr_cache:
                return vib_corr_cache[cache_key]

            vib_atoms = atoms.copy()
            vib_atoms.calc = DP(model=MODEL)
            vib_atoms.set_constraint(self._build_constraints(vib_atoms, with_internal_bonds=False))

            digest = hashlib.md5(str(cache_key).encode("utf-8")).hexdigest()[:10]
            vib_name = os.path.join(vib_root, f"{tag_prefix}_{digest}")
            g_corr, zpe, used_prefix = self._thermo_analysis(vib_atoms, temperature_K, name=vib_name)
            out = {
                "zpe": float(zpe),
                "g_corr": float(g_corr),
                "used_prefix": used_prefix,
                "has_vib": bool(used_prefix != 0),
            }
            vib_corr_cache[cache_key] = out
            return out

        results = []
        for rec in records:
            tag = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
            case_dir = os.path.join(out_root, tag)
            os.makedirs(case_dir, exist_ok=True)

            ts_atoms = read(rec["ts_xyz"])
            init_atoms = read(rec["initial_xyz"])
            rec_i, rec_j = rec["rec_bond_global"]

            ts_atoms.calc = DP(model=MODEL)
            ts_atoms.set_constraint(self._build_constraints(ts_atoms, with_internal_bonds=False))

            forward_traj = os.path.join(case_dir, "irc_forward.traj")
            reverse_traj = os.path.join(case_dir, "irc_reverse.traj")

            ts_fwd = read(rec["ts_xyz"])
            ts_fwd.calc = DP(model=MODEL)
            ts_fwd.set_constraint(self._build_constraints(ts_fwd, with_internal_bonds=False))
            irc_forward = IRC(
                ts_fwd,
                trajectory=forward_traj,
                dx=self.irc_dx,
                eta=self.irc_eta,
                ninner_iter=self.irc_ninner_iter,
                logfile=os.path.join(case_dir, "irc_forward.log"),
            )
            irc_forward.run(fmax=self.irc_fmax, steps=self.irc_steps, direction="forward")

            ts_rev = read(rec["ts_xyz"])
            ts_rev.calc = DP(model=MODEL)
            ts_rev.set_constraint(self._build_constraints(ts_rev, with_internal_bonds=False))
            irc_reverse = IRC(
                ts_rev,
                trajectory=reverse_traj,
                dx=self.irc_dx,
                eta=self.irc_eta,
                ninner_iter=self.irc_ninner_iter,
                logfile=os.path.join(case_dir, "irc_reverse.log"),
            )
            irc_reverse.run(fmax=self.irc_fmax, steps=self.irc_steps, direction="reverse")

            forward_path = read(forward_traj, index=":")
            reverse_path = read(reverse_traj, index=":")
            if len(forward_path) == 0 or len(reverse_path) == 0:
                print(f"IRC path empty for {tag}, skip.")
                continue

            reverse_final = reverse_path[-1]
            forward_final = forward_path[-1]

            reverse_final.calc = DP(model=MODEL)
            forward_final.calc = DP(model=MODEL)
            
            reverse_final.set_constraint(self._build_constraints(reverse_final, with_internal_bonds=False))
            forward_final.set_constraint(self._build_constraints(forward_final, with_internal_bonds=False))
            init_atoms.calc = DP(model=MODEL)
            init_atoms.set_constraint(self._build_constraints(init_atoms, with_internal_bonds=False))
            
            if os.path.exists(os.path.join(case_dir, "initial_opt.xyz")):
                init_atoms = read(os.path.join(case_dir, "initial_opt.xyz"))
                init_atoms.calc = DP(model=MODEL)
                init_atoms.set_constraint(self._build_constraints(init_atoms, with_internal_bonds=False))
            else:
                init_opt = QuasiNewton(init_atoms)
                try:
                    init_opt.run(steps=70, fmax=0.05)
                except ase.calculators.calculator.CalculationFailed:
                    print(f"Initial state optimization failed for {tag}; proceed with unoptimized initial state.")
                write(os.path.join(case_dir, "initial_opt.xyz"), init_atoms)

            _, dmat_init = get_distances(init_atoms.get_positions(), None, init_atoms.get_cell(), [True, True, True])
            d_break_init = float(dmat_init[rec_i, rec_j])

            _, dmat_r = get_distances(reverse_final.get_positions(), None, reverse_final.get_cell(), [True, True, True])
            _, dmat_f = get_distances(forward_final.get_positions(), None, forward_final.get_cell(), [True, True, True])
            d_break_r = float(dmat_r[rec_i, rec_j])
            d_break_f = float(dmat_f[rec_i, rec_j])

            z_i = int(ts_atoms[rec_i].number)
            z_j = int(ts_atoms[rec_j].number)
            r_sum = float(covalent_radii[z_i] + covalent_radii[z_j])
            # A conservative criterion for “clear dissociation” of the reactive bond.
            dissoc_threshold = max(1.35 * d_break_init, 1.25 * r_sum, d_break_init + 0.35)

            chosen_branch = "forward" if d_break_f >= d_break_r else "reverse"
            chosen_final = forward_final if chosen_branch == "forward" else reverse_final
            chosen_d_break = d_break_f if chosen_branch == "forward" else d_break_r

            used_perturbation_fallback = False
            perturbation_choice = None

            if max(d_break_f, d_break_r) < dissoc_threshold:
                ts_stem = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
                perturb_candidates = []
                perturb_specs = [
                    (
                        "imag_plus",
                        os.path.join(
                            self.path,
                            "rxn",
                            "TS_guesses",
                            f"{ts_stem}_imag_plus.xyz",
                        ),
                    ),
                    (
                        "imag_minus",
                        os.path.join(
                            self.path,
                            "rxn",
                            "TS_guesses",
                            f"{ts_stem}_imag_minus.xyz",
                        ),
                    ),
                ]

                for label, path in perturb_specs:
                    if not os.path.exists(path):
                        continue
                    try:
                        cand = read(path)
                        cand.calc = DP(model=MODEL)
                        cand.set_constraint(self._build_constraints(cand, with_internal_bonds=False))
                        _, dmat_c = get_distances(
                            cand.get_positions(), None, cand.get_cell(), [True, True, True]
                        )
                        d_break_c = float(dmat_c[rec_i, rec_j])
                        e_c = float(cand.get_potential_energy())
                        perturb_candidates.append({
                            "label": label,
                            "path": path,
                            "atoms": cand,
                            "d_break": d_break_c,
                            "energy": e_c,
                        })
                    except Exception:
                        continue

                if perturb_candidates:
                    dissoc_like = [c for c in perturb_candidates if c["d_break"] >= dissoc_threshold]
                    if dissoc_like:
                        dissoc_like.sort(key=lambda x: (x["d_break"], -x["energy"]), reverse=True)
                        best_pert = dissoc_like[0]
                    else:
                        perturb_candidates.sort(key=lambda x: (x["d_break"], -x["energy"]), reverse=True)
                        best_pert = perturb_candidates[0]

                    if best_pert["d_break"] > chosen_d_break + 0.05:
                        chosen_final = best_pert["atoms"]
                        chosen_branch = f"perturbation_{best_pert['label']}"
                        chosen_d_break = float(best_pert["d_break"])
                        used_perturbation_fallback = True
                        perturbation_choice = {
                            "label": best_pert["label"],
                            "source_xyz": os.path.abspath(best_pert["path"]),
                            "d_break": float(best_pert["d_break"]),
                            "energy": float(best_pert["energy"]),
                        }
                        write(os.path.join(case_dir, "final_perturbation.xyz"), chosen_final)
                        print(
                            f"{tag}: IRC final states are not clearly dissociated; "
                            f"fallback to {best_pert['label']} perturbation endpoint."
                        )

            write(os.path.join(case_dir, "initial.xyz"), init_atoms)
            write(os.path.join(case_dir, "ts.xyz"), ts_atoms)
            write(os.path.join(case_dir, "final_forward.xyz"), forward_final)
            write(os.path.join(case_dir, "final_reverse.xyz"), reverse_final)
            write(os.path.join(case_dir, "final_selected.xyz"), chosen_final)

            reverse_rev = list(reverse_path)
            reverse_rev.reverse()
            full_path = reverse_rev + forward_path[1:]
            write(os.path.join(case_dir, "irc_full_path.xyz"), full_path)

            result = {
                "tag": tag,
                "temperature_K": float(temperature_K),
                "irc_thermo_corrections_enabled": bool(enable_irc_thermo),
                "initial_xyz": os.path.abspath(os.path.join(case_dir, "initial.xyz")),
                "ts_xyz": os.path.abspath(os.path.join(case_dir, "ts.xyz")),
                "final_forward_xyz": os.path.abspath(os.path.join(case_dir, "final_forward.xyz")),
                "final_reverse_xyz": os.path.abspath(os.path.join(case_dir, "final_reverse.xyz")),
                "final_selected_xyz": os.path.abspath(os.path.join(case_dir, "final_selected.xyz")),
                "chosen_branch": chosen_branch,
                "broken_bond_global": [int(rec_i), int(rec_j)],
                "d_break_initial": d_break_init,
                "dissociation_threshold": dissoc_threshold,
                "d_break_forward": d_break_f,
                "d_break_reverse": d_break_r,
                "d_break_selected": chosen_d_break,
                "e_forward": float(forward_final.get_potential_energy()),
                "e_reverse": float(reverse_final.get_potential_energy()),
                "e_selected": float(chosen_final.get_potential_energy()),
                "e_initial": float(init_atoms.get_potential_energy()),
                "e_ts": float(ts_atoms.get_potential_energy()),
                "used_perturbation_fallback": bool(used_perturbation_fallback),
                "perturbation_choice": perturbation_choice,
            }

            init_vib = _vib_corrections_for_atoms(init_atoms, result["initial_xyz"], f"{tag}_is")
            ts_vib = _vib_corrections_for_atoms(ts_atoms, result["ts_xyz"], f"{tag}_ts")
            # forward_vib = _vib_corrections_for_atoms(forward_final, result["final_forward_xyz"], f"{tag}_fwd")
            # reverse_vib = _vib_corrections_for_atoms(reverse_final, result["final_reverse_xyz"], f"{tag}_rev")
            selected_vib = _vib_corrections_for_atoms(
                chosen_final,
                f"{tag}::selected::{chosen_branch}",
                f"{tag}_selected",
            )

            result.update({
                "zpe_correction_initial": float(init_vib["zpe"]),
                "zpe_correction_ts": float(ts_vib["zpe"]),
                # "zpe_correction_forward": float(forward_vib["zpe"]),
                # "zpe_correction_reverse": float(reverse_vib["zpe"]),
                "zpe_correction_selected": float(selected_vib["zpe"]),
                "g_correction_initial": float(init_vib["g_corr"]),
                "g_correction_ts": float(ts_vib["g_corr"]),
                # "g_correction_forward": float(forward_vib["g_corr"]),
                # "g_correction_reverse": float(reverse_vib["g_corr"]),
                "g_correction_selected": float(selected_vib["g_corr"]),
                "e_initial_zpe": float(result["e_initial"] + init_vib["zpe"]),
                "e_ts_zpe": float(result["e_ts"] + ts_vib["zpe"]),
                # "e_forward_zpe": float(result["e_forward"] + forward_vib["zpe"]),
                # "e_reverse_zpe": float(result["e_reverse"] + reverse_vib["zpe"]),
                "e_selected_zpe": float(result["e_selected"] + selected_vib["zpe"]),
                "g_initial": float(result["e_initial"] + init_vib["g_corr"]),
                "g_ts": float(result["e_ts"] + ts_vib["g_corr"]),
                # "g_forward": float(result["e_forward"] + forward_vib["g_corr"]),
                # "g_reverse": float(result["e_reverse"] + reverse_vib["g_corr"]),
                "g_selected": float(result["e_selected"] + selected_vib["g_corr"]),
                "barrier_e_ts_minus_initial_zpe": float((result["e_ts"] + ts_vib["zpe"]) - (result["e_initial"] + init_vib["zpe"])),
                "delta_e_selected_minus_initial_zpe": float((result["e_selected"] + selected_vib["zpe"]) - (result["e_initial"] + init_vib["zpe"])),
                "barrier_g_ts_minus_initial": float((result["e_ts"] + ts_vib["g_corr"]) - (result["e_initial"] + init_vib["g_corr"])),
                "delta_g_selected_minus_initial": float((result["e_selected"] + selected_vib["g_corr"]) - (result["e_initial"] + init_vib["g_corr"])),
                "g_zpe_is": float(result["e_initial"] + init_vib["g_corr"] + init_vib["zpe"]),
                "g_zpe_ts": float(result["e_ts"] + ts_vib["g_corr"] + ts_vib["zpe"]),
                "g_zpe_fs": float(result["e_selected"] + selected_vib["g_corr"] + selected_vib["zpe"]),
                "barrier_g_zpe_ts_minus_is": float((result["e_ts"] + ts_vib["g_corr"] + ts_vib["zpe"]) - (result["e_initial"] + init_vib["g_corr"] + init_vib["zpe"])),
                "delta_g_zpe_selected_minus_is": float((result["e_selected"] + selected_vib["g_corr"] + selected_vib["zpe"]) - (result["e_initial"] + init_vib["g_corr"] + init_vib["zpe"])),
                "barrier_g_zpe_ts_minus_fs": float((result["e_ts"] + ts_vib["g_corr"] + ts_vib["zpe"]) - (result["e_selected"] + selected_vib["g_corr"] + selected_vib["zpe"])),
                "vib_source_initial": init_vib["used_prefix"],
                "vib_source_ts": ts_vib["used_prefix"],
                # "vib_source_forward": forward_vib["used_prefix"],
                # "vib_source_reverse": reverse_vib["used_prefix"],
                "vib_source_selected": selected_vib["used_prefix"],
            })

            result["e_ts_zpe"] = float(result["e_ts"] + ts_vib["zpe"])
            result["g_ts"] = float(result["e_ts"] + ts_vib["g_corr"])

            with open(os.path.join(case_dir, "irc_summary.yaml"), "w") as f:
                yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
            results.append(result)

        result_path = os.path.join(out_root, "irc_results.yaml")
        with open(result_path, "w") as f:
            yaml.safe_dump({"irc_results": results}, f, sort_keys=False, allow_unicode=True)

        print(f"IRC final-state search finished: {len(results)} cases. Summary: {result_path}")
        return results
    
    def get_final_state_energy(self):
        records = self._load_ts_records()
        if len(records) == 0:
            print("No TS records found; skip final-state energy evaluation.")
            return []

        out_root = os.path.join(self.path, "rxn", "FS_energy")
        os.makedirs(out_root, exist_ok=True)
        vib_root = os.path.join(out_root, "vib_jobs")
        if self.enable_thermo_corrections:
            os.makedirs(vib_root, exist_ok=True)
        temperature_K = self.temperature

        if not self.enable_thermo_corrections:
            print("Thermo corrections are disabled; all ZPE/G corrections will be 0.")

        def _energy_from_xyz(xyz_path):
            xyz_path = os.path.abspath(xyz_path)
            if xyz_path in energy_cache:
                return energy_cache[xyz_path]
            atoms = read(xyz_path)
            atoms.calc = DP(model=MODEL)
            atoms.set_constraint(self._build_constraints(atoms, with_internal_bonds=False))
            e = float(atoms.get_potential_energy())
            energy_cache[xyz_path] = e
            return e

        def _energy_from_gas_species(sp_id):
            """Evaluate isolated-gas energy for one species template in a large vacuum box."""
            cache_key = f"gas::{sp_id}"
            if cache_key in energy_cache:
                return energy_cache[cache_key]

            if sp_id not in self.ads_templates:
                raise KeyError(f"Unknown species id for gas energy: {sp_id}")

            gas_atoms = self.ads_templates[sp_id]["atoms"].copy()
            gas_atoms.set_cell([20.0, 20.0, 20.0])
            gas_atoms.set_pbc([False, False, False])
            gas_atoms.center()
            gas_atoms.calc = DP(model=MODEL)
            e = float(gas_atoms.get_potential_energy())
            energy_cache[cache_key] = e
            return e

        energy_cache = {}
        vib_corr_cache = {}

        def _vib_corrections_for_structure(xyz_path):
            cache_key = os.path.abspath(xyz_path)
            if cache_key in vib_corr_cache:
                return vib_corr_cache[cache_key]

            if not self.enable_thermo_corrections:
                out = {
                    "zpe": 0.0,
                    "g_corr": 0.0,
                    "used_prefix": 0,
                    "has_vib": False,
                }
                vib_corr_cache[cache_key] = out
                return out

            atoms = read(xyz_path)
            atoms.calc = DP(model=MODEL)
            atoms.set_constraint(self._build_constraints(atoms, with_internal_bonds=False))

            digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
            base = os.path.splitext(os.path.basename(cache_key))[0]
            vib_name = os.path.join(vib_root, f"{base}_{digest}")

            g_corr, zpe, used_prefix = self._thermo_analysis(atoms, temperature_K, name=vib_name)
            out = {
                "zpe": float(zpe),
                "g_corr": float(g_corr),
                "used_prefix": used_prefix,
                "has_vib": bool(used_prefix != 0),
            }
            vib_corr_cache[cache_key] = out
            return out

        def _find_adsorbate_energy_file(sp_id, site):
            stem_tagged = self._tagged_stem(f"{site}")
            for ads_dir in self._adsorbate_dir_candidates(sp_id):
                candidates = [
                    os.path.join(ads_dir, f"{stem_tagged}_opt.xyz"),
                    # os.path.join(ads_dir, f"{stem_tagged}_weakopt.xyz"),
                    # os.path.join(ads_dir, f"{stem_tagged}.xyz"),
                    os.path.join(ads_dir, f"{site}_opt.xyz"),
                    # os.path.join(ads_dir, f"{site}_weakopt.xyz"),
                    # os.path.join(ads_dir, f"{site}.xyz"),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        return cand
            return None

        def _find_adsorbate_energy_file_global_min(sp_id):
            """For far-separated FS: prefer pre-ranked best site from generate_initial_adsorbate_guesses."""
            valid_sites = self.valid_sites.get(sp_id, []) if hasattr(self, "valid_sites") else []
            if len(valid_sites) > 0:
                best_site = valid_sites[0]
                best_path = _find_adsorbate_energy_file(sp_id, best_site)
                if best_path is not None:
                    return best_path

            # Fallback: scan all optimized structures if valid_sites is unavailable
            # or the expected best-site file is missing.
            opt_files = []
            for ads_dir in self._adsorbate_dir_candidates(sp_id):
                if not os.path.isdir(ads_dir):
                    continue
                opt_files.extend([
                    os.path.join(ads_dir, name)
                    for name in os.listdir(ads_dir)
                    if name.endswith("_opt.xyz")
                ])
            if len(opt_files) == 0:
                return None

            best_path = None
            best_energy = None
            for xyz_path in opt_files:
                intact, reason = self._is_adsorbate_structure_intact(xyz_path, sp_id)
                # if not intact:
                #     print(f"Skip dissociated adsorbate candidate for {sp_id}: {xyz_path} ({reason})")
                #     continue
                try:
                    e_ads = _energy_from_xyz(xyz_path)
                except Exception:
                    continue
                if (best_energy is None) or (e_ads < best_energy):
                    best_energy = e_ads
                    best_path = xyz_path
            return best_path

        def _collect_state_items(species_ids, site, tag, state_name, far_separated=False):
            items = []
            for sp_id in species_ids:
                if far_separated:
                    xyz_path = _find_adsorbate_energy_file_global_min(sp_id)
                else:
                    xyz_path = _find_adsorbate_energy_file(sp_id, site)
                if xyz_path is None:
                    if far_separated:
                        print(f"Skip {tag}: missing {state_name} adsorbate file for {sp_id} over all sites")
                    else:
                        print(f"Skip {tag}: missing {state_name} adsorbate file for {sp_id} at site {site}")
                    return None
                try:
                    e_ads = _energy_from_xyz(xyz_path)
                except Exception as exc:
                    print(f"Skip {tag}: failed to evaluate {xyz_path}: {exc}")
                    return None

                vib_corr = _vib_corrections_for_structure(xyz_path)

                items.append({
                    "species": sp_id,
                    "xyz": os.path.abspath(xyz_path),
                    "energy": float(e_ads),
                    "zpe_correction": float(vib_corr["zpe"]),
                    "g_correction": float(vib_corr["g_corr"]),
                    "vib_source": vib_corr["used_prefix"],
                })
            return items

        def _state_energy_from_items(items, e_slab_local):
            # Build total energy of a multi-fragment adsorbate state from
            # single-adsorbate-on-slab energies by removing repeated slab terms.
            n_items = len(items)
            e_sum = float(sum(x["energy"] for x in items))
            return float(e_sum - max(0, n_items - 1) * e_slab_local)

        def _state_correction_from_items(items, key, slab_corr=0.0):
            # Keep correction assembly consistent with state energy assembly:
            # E_state = sum(E_ads_on_slab) - (n_items-1) * E_slab.
            n_items = len(items)
            corr_sum = float(sum(float(x.get(key, 0.0)) for x in items))
            return float(corr_sum - max(0, n_items - 1) * float(slab_corr))

        slab_opt_xyz = os.path.join(out_root, self._tagged_stem("slab_opt") + ".xyz")
        if os.path.exists(slab_opt_xyz):
            slab_atoms = read(slab_opt_xyz)
            slab_atoms.calc = DP(model=MODEL)
            slab_atoms.set_constraint(self._build_constraints(slab_atoms, with_internal_bonds=False))
            e_slab = float(slab_atoms.get_potential_energy())
            print(f"Reuse optimized slab: {slab_opt_xyz}")
        else:
            slab_atoms = self.slab.stru.copy()
            slab_atoms.calc = DP(model=MODEL)
            slab_atoms.set_constraint(self._build_constraints(slab_atoms, with_internal_bonds=False))
            slab_optimizer = QuasiNewton(slab_atoms)
            slab_optimizer.run(fmax=0.05, steps=100)
            e_slab = float(slab_atoms.get_potential_energy())
            write(slab_opt_xyz, slab_atoms)

        slab_xyz_for_vib = os.path.join(out_root, self._tagged_stem("slab_for_vib") + ".xyz")
        write(slab_xyz_for_vib, slab_atoms)
        slab_vib_corr = _vib_corrections_for_structure(slab_xyz_for_vib)

        results = []
        seen_tags = set()
        for rec in records:
            tag = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
            if tag in seen_tags:
                continue
            seen_tags.add(tag)

            rxn_idx = self._resolve_record_rxn_index(rec)
            site = int(rec["site"])
            if rxn_idx is None or rxn_idx < 0 or rxn_idx >= len(self.rxns_dict):
                print(f"Skip {tag}: cannot map record to current reaction list")
                continue

            rxn = self.rxns_dict[rxn_idx]
            reactant_species = list(rxn.get("reactant_species", []))
            product_species = list(rxn.get("product_species", []))
            if len(reactant_species) == 0:
                print(f"Skip {tag}: reactant_species is empty")
                continue
            if len(product_species) < 2:
                print(f"Skip {tag}: product_species count < 2 ({product_species})")
                continue

            is_items = _collect_state_items(reactant_species, site, tag, "IS", 
                                            far_separated=True)
            if is_items is None:
                continue

            # FS is treated as far-separated products (ignore migration barrier):
            # each product uses its own lowest-energy adsorption structure across sites.
            fs_items = _collect_state_items(
                product_species[:2],
                site,
                tag,
                "FS",
                far_separated=True,
            )
            if fs_items is None:
                continue

            ts_xyz_path = rec.get("ts_xyz")
            if not ts_xyz_path or (not os.path.exists(ts_xyz_path)):
                print(f"Skip {tag}: missing TS xyz file: {ts_xyz_path}")
                continue
            try:
                e_ts = _energy_from_xyz(ts_xyz_path)
            except Exception as exc:
                print(f"Skip {tag}: failed to evaluate TS energy from {ts_xyz_path}: {exc}")
                continue

            ts_vib_corr = _vib_corrections_for_structure(ts_xyz_path)

            e_is = _state_energy_from_items(is_items, e_slab)
            e_fs = _state_energy_from_items(fs_items, e_slab)
            e_a_is = float(e_ts - e_is)
            e_a_fs = float(e_ts - e_fs)
            e_fs_is = float(e_fs - e_is)

            zpe_is_corr = _state_correction_from_items(
                is_items,
                "zpe_correction",
                slab_corr=slab_vib_corr["zpe"],
            )
            zpe_fs_corr = _state_correction_from_items(
                fs_items,
                "zpe_correction",
                slab_corr=slab_vib_corr["zpe"],
            )
            zpe_ts_corr = float(ts_vib_corr["zpe"])

            g_is_corr = _state_correction_from_items(
                is_items,
                "g_correction",
                slab_corr=slab_vib_corr["g_corr"],
            )
            g_fs_corr = _state_correction_from_items(
                fs_items,
                "g_correction",
                slab_corr=slab_vib_corr["g_corr"],
            )
            g_ts_corr = float(ts_vib_corr["g_corr"])

            e_is_zpe = float(e_is + zpe_is_corr)
            e_ts_zpe = float(e_ts + zpe_ts_corr)
            e_fs_zpe = float(e_fs + zpe_fs_corr)

            g_is = float(e_is + g_is_corr)
            g_ts = float(e_ts + g_ts_corr)
            g_fs = float(e_fs + g_fs_corr)

            barrier_ts_minus_is_zpe = float(e_ts_zpe - e_is_zpe)
            barrier_ts_minus_fs_zpe = float(e_ts_zpe - e_fs_zpe)
            delta_e_fs_minus_is_zpe = float(e_fs_zpe - e_is_zpe)

            barrier_g_ts_minus_is = float(g_ts - g_is)
            barrier_g_ts_minus_fs = float(g_ts - g_fs)
            delta_g_fs_minus_is = float(g_fs - g_is)

            g_zpe_is = float(e_is + zpe_is_corr + g_is_corr)
            g_zpe_ts = float(e_ts + zpe_ts_corr + g_ts_corr)
            g_zpe_fs = float(e_fs + zpe_fs_corr + g_fs_corr)
            barrier_g_zpe_ts_minus_is = float(g_zpe_ts - g_zpe_is)
            barrier_g_zpe_ts_minus_fs = float(g_zpe_ts - g_zpe_fs)
            delta_g_zpe_fs_minus_is = float(g_zpe_fs - g_zpe_is)

            row = {
                "tag": tag,
                "rxn_idx": rxn_idx,
                "site": site,
                "reactants": is_items,
                "products": fs_items,
                "temperature_K": float(temperature_K),
                "e_slab": e_slab,
                "e_is": e_is,
                "e_ts": float(e_ts),
                "e_fs": e_fs,
                "barrier_ts_minus_is": e_a_is,
                "barrier_ts_minus_fs": e_a_fs,
                "delta_e_fs_minus_is": e_fs_is,
                "zpe_correction_is": zpe_is_corr,
                "zpe_correction_ts": zpe_ts_corr,
                "zpe_correction_fs": zpe_fs_corr,
                "e_is_zpe": e_is_zpe,
                "e_ts_zpe": e_ts_zpe,
                "e_fs_zpe": e_fs_zpe,
                "barrier_ts_minus_is_zpe": barrier_ts_minus_is_zpe,
                "barrier_ts_minus_fs_zpe": barrier_ts_minus_fs_zpe,
                "delta_e_fs_minus_is_zpe": delta_e_fs_minus_is_zpe,
                "g_correction_is": g_is_corr,
                "g_correction_ts": g_ts_corr,
                "g_correction_fs": g_fs_corr,
                "g_is": g_is,
                "g_ts": g_ts,
                "g_fs": g_fs,
                "barrier_g_ts_minus_is": barrier_g_ts_minus_is,
                "barrier_g_ts_minus_fs": barrier_g_ts_minus_fs,
                "delta_g_fs_minus_is": delta_g_fs_minus_is,
                "g_zpe_is": g_zpe_is,
                "g_zpe_ts": g_zpe_ts,
                "g_zpe_fs": g_zpe_fs,
                "barrier_g_zpe_ts_minus_is": barrier_g_zpe_ts_minus_is,
                "barrier_g_zpe_ts_minus_fs": barrier_g_zpe_ts_minus_fs,
                "delta_g_zpe_fs_minus_is": delta_g_zpe_fs_minus_is,
                "slab_zpe_correction": float(slab_vib_corr["zpe"]),
                "slab_g_correction": float(slab_vib_corr["g_corr"]),
                "slab_vib_source": slab_vib_corr["used_prefix"],
                "ts_vib_source": ts_vib_corr["used_prefix"],
                "ts_xyz": os.path.abspath(ts_xyz_path),
            }
            results.append(row)

        results_path = os.path.join(out_root, self._tagged_stem("final_state_energy") + ".yaml")
        with open(results_path, "w") as f:
            yaml.safe_dump({"final_state_energy": results}, f, sort_keys=False, allow_unicode=True)

        print(
            "Final-state energy evaluation finished: "
            f"{len(results)} cases. Summary: {results_path}"
        )
        return results

    def get_ads_energy(self):
        out_root = os.path.join(self.path, "rxn", "FS_energy")
        os.makedirs(out_root, exist_ok=True)
        vib_root = os.path.join(out_root, "vib_jobs")
        if self.enable_thermo_corrections:
            os.makedirs(vib_root, exist_ok=True)
        temperature_K = self.temperature

        if not self.enable_thermo_corrections:
            print("Thermo corrections are disabled; all ZPE/G corrections will be 0.")

        energy_cache = {}
        vib_corr_cache = {}

        def _energy_from_xyz(xyz_path):
            xyz_path = os.path.abspath(xyz_path)
            if xyz_path in energy_cache:
                return energy_cache[xyz_path]
            atoms = read(xyz_path)
            atoms.calc = DP(model=MODEL)
            atoms.set_constraint(self._build_constraints(atoms, with_internal_bonds=False))
            e = float(atoms.get_potential_energy())
            energy_cache[xyz_path] = e
            return e

        def _energy_from_gas_species(sp_id):
            cache_key = f"gas::{sp_id}"
            if cache_key in energy_cache:
                return energy_cache[cache_key]
            if sp_id not in self.ads_templates:
                raise KeyError(f"Unknown species id for gas energy: {sp_id}")
            gas_atoms = self.ads_templates[sp_id]["atoms"].copy()
            gas_atoms.set_cell([20.0, 20.0, 20.0])
            gas_atoms.set_pbc([False, False, False])
            gas_atoms.center()
            gas_atoms.calc = DP(model=MODEL)
            e = float(gas_atoms.get_potential_energy())
            energy_cache[cache_key] = e
            return e

        def _vib_corrections_for_structure(xyz_path):
            cache_key = os.path.abspath(xyz_path)
            if cache_key in vib_corr_cache:
                return vib_corr_cache[cache_key]
            if not self.enable_thermo_corrections:
                out = {"zpe": 0.0, "g_corr": 0.0, "used_prefix": 0, "has_vib": False}
                vib_corr_cache[cache_key] = out
                return out

            atoms = read(xyz_path)
            atoms.calc = DP(model=MODEL)
            atoms.set_constraint(self._build_constraints(atoms, with_internal_bonds=False))
            digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
            base = os.path.splitext(os.path.basename(cache_key))[0]
            vib_name = os.path.join(vib_root, f"{base}_{digest}")
            g_corr, zpe, used_prefix = self._thermo_analysis(atoms, temperature_K, name=vib_name)
            out = {
                "zpe": float(zpe),
                "g_corr": float(g_corr),
                "used_prefix": used_prefix,
                "has_vib": bool(used_prefix != 0),
            }
            vib_corr_cache[cache_key] = out
            return out

        def _vib_corrections_for_gas_species(sp_id):
            cache_key = f"gas::{sp_id}"
            if cache_key in vib_corr_cache:
                return vib_corr_cache[cache_key]
            if not self.enable_thermo_corrections:
                out = {"zpe": 0.0, "g_corr": 0.0, "used_prefix": 0, "has_vib": False}
                vib_corr_cache[cache_key] = out
                return out

            gas_atoms = self.ads_templates[sp_id]["atoms"].copy()
            gas_atoms.set_cell([20.0, 20.0, 20.0])
            gas_atoms.set_pbc([False, False, False])
            gas_atoms.center()
            gas_atoms.calc = DP(model=MODEL)
            digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
            vib_name = os.path.join(vib_root, f"gas_{sp_id}_{digest}")
            g_corr, zpe, used_prefix = self._gas_ideal_thermo_corrections(
                gas_atoms,
                temperature_K=temperature_K,
                pressure_pa=self.gas_pressure_pa,
                name=vib_name,
                indices=list(range(len(gas_atoms))),
            )
            out = {
                "zpe": float(zpe),
                "g_corr": float(g_corr),
                "used_prefix": used_prefix,
                "has_vib": bool(used_prefix != 0),
            }
            vib_corr_cache[cache_key] = out
            return out

        def _find_adsorbate_energy_file(sp_id, site):
            stem_tagged = self._tagged_stem(f"{site}")
            for ads_dir in self._adsorbate_dir_candidates(sp_id):
                candidates = [
                    os.path.join(ads_dir, f"{stem_tagged}_opt.xyz"),
                    os.path.join(ads_dir, f"{site}_opt.xyz"),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        return cand
            return None

        def _find_adsorbate_energy_file_global_min(sp_id):
            valid_sites = self.valid_sites.get(sp_id, []) if hasattr(self, "valid_sites") else []
            if len(valid_sites) > 0:
                best_path = _find_adsorbate_energy_file(sp_id, valid_sites[0])
                if best_path is not None:
                    return best_path

            opt_files = []
            for ads_dir in self._adsorbate_dir_candidates(sp_id):
                if not os.path.isdir(ads_dir):
                    continue
                opt_files.extend([
                    os.path.join(ads_dir, name)
                    for name in os.listdir(ads_dir)
                    if name.endswith("_opt.xyz")
                ])
            if len(opt_files) == 0:
                return None

            best_path = None
            best_energy = None
            for xyz_path in opt_files:
                intact, reason = self._is_adsorbate_structure_intact(xyz_path, sp_id)
                # if not intact:
                #     print(f"Skip dissociated adsorbate candidate for {sp_id}: {xyz_path} ({reason})")
                #     continue
                try:
                    e_ads = _energy_from_xyz(xyz_path)
                except Exception:
                    continue
                if (best_energy is None) or (e_ads < best_energy):
                    best_energy = e_ads
                    best_path = xyz_path
            return best_path

        slab_opt_xyz = os.path.join(out_root, self._tagged_stem("slab_opt") + ".xyz")
        if os.path.exists(slab_opt_xyz):
            slab_atoms = read(slab_opt_xyz)
            slab_atoms.calc = DP(model=MODEL)
            slab_atoms.set_constraint(self._build_constraints(slab_atoms, with_internal_bonds=False))
            e_slab = float(slab_atoms.get_potential_energy())
        else:
            slab_atoms = self.slab.stru.copy()
            slab_atoms.calc = DP(model=MODEL)
            slab_atoms.set_constraint(self._build_constraints(slab_atoms, with_internal_bonds=False))
            slab_optimizer = QuasiNewton(slab_atoms)
            slab_optimizer.run(fmax=0.05, steps=100)
            e_slab = float(slab_atoms.get_potential_energy())
            write(slab_opt_xyz, slab_atoms)

        slab_xyz_for_vib = os.path.join(out_root, self._tagged_stem("slab_for_vib") + ".xyz")
        write(slab_xyz_for_vib, slab_atoms)
        slab_vib_corr = _vib_corrections_for_structure(slab_xyz_for_vib)

        gas_capable_species = []
        if self.gas_species_whitelist is not None and len(self.gas_species_whitelist) > 0:
            for sp_id in self.gas_species_whitelist:
                if sp_id not in self.ads_templates:
                    print(f"Skip whitelist species {sp_id}: not found in ads templates")
                    continue
                gas_capable_species.append(sp_id)

        for sp_id, info in self.ads_templates.items():
            if len(info.get("ad_idx", [])) == 0 and sp_id not in gas_capable_species:
                gas_capable_species.append(sp_id)
        # 如果是CO则强制加入（因为CO的气相能量很重要，且通常会有优化结构）
        for sp_id, info in self.ads_templates.items():
            sp_name = str(info.get("name", "")).strip().upper()
            sp_key = str(sp_id).strip().upper()
            if (sp_key == "CO" or sp_name == "CO") and sp_id not in gas_capable_species:
                gas_capable_species.append(sp_id)

        gas_capable_species = cfgmod._dedupe_keep_order(gas_capable_species)

        adsorption_results = []
        for sp_id in gas_capable_species:
            ads_xyz_path = _find_adsorbate_energy_file_global_min(sp_id)
            if ads_xyz_path is None:
                print(f"Skip adsorption energy for {sp_id}: no optimized adsorbate structure found")
                continue
            try:
                e_ads_on_slab = _energy_from_xyz(ads_xyz_path)
                e_gas = _energy_from_gas_species(sp_id)
            except Exception as exc:
                print(f"Skip adsorption energy for {sp_id}: energy evaluation failed: {exc}")
                continue

            ads_vib = _vib_corrections_for_structure(ads_xyz_path)
            gas_vib = _vib_corrections_for_gas_species(sp_id)
            delta_e_ads = float(e_ads_on_slab - e_slab - e_gas)
            delta_zpe_ads = float(ads_vib["zpe"] - slab_vib_corr["zpe"] - gas_vib["zpe"])
            delta_g_corr_ads = float(ads_vib["g_corr"] - slab_vib_corr["g_corr"] - gas_vib["g_corr"])

            adsorption_results.append({
                "species": sp_id,
                "species_name": self.ads_templates[sp_id].get("name", sp_id),
                "adsorbate_xyz": os.path.abspath(ads_xyz_path),
                "temperature_K": float(temperature_K),
                "e_ads_on_slab": float(e_ads_on_slab),
                "e_slab": float(e_slab),
                "e_gas": float(e_gas),
                "adsorption_energy": delta_e_ads,
                "delta_zpe_adsorption": delta_zpe_ads,
                "delta_g_correction_adsorption": delta_g_corr_ads,
                "adsorption_energy_zpe": float(delta_e_ads + delta_zpe_ads),
                "adsorption_free_energy": float(delta_e_ads + delta_g_corr_ads),
                "adsorption_free_energy_zpe": float(delta_e_ads + delta_zpe_ads + delta_g_corr_ads),
                "ads_vib_source": ads_vib["used_prefix"],
                "slab_vib_source": slab_vib_corr["used_prefix"],
                "gas_vib_source": gas_vib["used_prefix"],
            })

        adsorption_results_path = os.path.join(out_root, self._tagged_stem("adsorption_energy") + ".yaml")
        with open(adsorption_results_path, "w") as f:
            yaml.safe_dump({"adsorption_energy": adsorption_results}, f, sort_keys=False, allow_unicode=True)

        print(
            "Adsorption-energy evaluation finished: "
            f"{len(adsorption_results)} species. Summary: {adsorption_results_path}"
        )
        return adsorption_results

from utils.workflow import DPWorkflow


if __name__ == "__main__":
    parser = cfgmod.build_arg_parser()
    args = parser.parse_args()
    cfg = cfgmod.config_from_args(args)
    slab_entries = cfgmod.resolve_slab_paths_for_workflow(cfg)

    for entry in slab_entries:
        group_index = entry["group_index"]
        slab_path_for_run = entry["slab_path"]
        output_suffix = f"vg{group_index}" if group_index is not None else ""

        if output_suffix:
            print(f"\n===== Start workflow for vacancy group {group_index} =====")
        else:
            print("\n===== Start workflow =====")

        workflow = DPWorkflow(
            path=cfg.path,
            prepared_data_file=cfg.prepared,
            slab_path=slab_path_for_run,
            top_x=cfg.top_x,
            enable_rotation_enum_ads=cfg.enum_ads,
            enable_rotation_enum_ts=cfg.enum_ts,
            surface_normal=cfg.surface_normal,
            normal_axis=cfg.normal_axis,
            enable_imag_mode_check=cfg.imag_mode_check,
            imag_mode_displacement=cfg.imag_mode_displacement,
            imag_mode_relax_steps=cfg.imag_mode_relax_steps,
            bottom_freeze_threshold=cfg.bottom_freeze_threshold,
            run_irc_final_state=cfg.run_irc_final_state,
            irc_fmax=cfg.irc_fmax,
            irc_steps=cfg.irc_steps,
            irc_dx=cfg.irc_dx,
            irc_eta=cfg.irc_eta,
            irc_ninner_iter=cfg.irc_ninner_iter,
            enable_irc_thermo_corrections=cfg.irc_thermo_corrections,
            irc_temperature=cfg.irc_temperature,
            enable_thermo_corrections=cfg.thermo_corrections,
            gas_species_whitelist=cfg.gas_species_whitelist,
            gas_pressure_pa=cfg.gas_pressure_pa,
            output_suffix=output_suffix,
        )
        workflow.generate_initial_adsorbate_guesses()
        workflow.generate_rxn_ts_guesses_ccqn()
        if cfg.run_irc_final_state:
            workflow.get_final_state_IRC()
        workflow.get_final_state_energy()
        workflow.get_ads_energy()

        if output_suffix:
            print(f"===== Finished workflow for vacancy group {group_index} =====\n")
        else:
            print("===== Finished workflow =====\n")
