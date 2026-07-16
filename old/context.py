import os
import yaml
import hashlib
import json
from ase.io import read
import numpy as np
import shutil
from slabsite import SlabSite
from utils import config as cfgmod
from utils import geometry as geom

class WorkflowContext:
    """共享所有工作流模块的数据和配置"""
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

    # ---------- 工具方法 ----------
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
