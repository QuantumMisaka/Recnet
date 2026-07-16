import os
import numpy as np
import shutil
from ase.io import read, write
from ase.constraints import FixAtoms, FixInternals
from ase.optimize import MDMin, QuasiNewton
from ase.calculators import calculator
from ase.geometry import get_distances
from ase.data import covalent_radii
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from deepmd.calculator import DP
from utils import geometry as geom
from utils import constraints as constraint_utils

MODEL = "/data/home/youyinglong/model/dpa230-v2-simp/FeCHO-dpa231-v2-7-3heads-100w.pth"

def frozen_indices(atoms, normal_axis, bottom_freeze_threshold):
    axis_id = {"x": 0, "y": 1, "z": 2}[normal_axis]
    return [atom.index for atom in atoms if atom.position[axis_id] < bottom_freeze_threshold]

def build_constraints(atoms, normal_axis, bottom_freeze_threshold, surf_atom_num=0, with_internal_bonds=False):
    constraints = [FixAtoms(indices=frozen_indices(atoms, normal_axis, bottom_freeze_threshold))]
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

def site_lift_vector(surface_normal, adsorbate_lift=0.8):
    return np.array(surface_normal, dtype=float) * adsorbate_lift

def adsorbate_bond_set(atoms):
    bonds = set()
    for i, j in geom.get_bond_connections(atoms, shift=0, cutoff=1.2, bond_type="nosurf"):
        a, b = int(i), int(j)
        bonds.add((a, b) if a <= b else (b, a))
    return bonds

def find_heavy_neighbor(stru, atom_idx, exclude_idx=None, cutoff_scale=1.25):
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

def is_adsorbate_structure_intact(xyz_path, sp_id, ads_templates, slab_atom_count):
    if sp_id not in ads_templates:
        return False, "unknown_species"
    try:
        full_atoms = read(xyz_path)
    except Exception as exc:
        return False, f"read_failed: {exc}"
    template_atoms = ads_templates[sp_id]["atoms"]
    n_ads = len(template_atoms)
    if len(full_atoms) < slab_atom_count + n_ads:
        return False, "atom_count_mismatch"
    ads_atoms = full_atoms[slab_atom_count: slab_atom_count + n_ads]
    template_bonds = adsorbate_bond_set(template_atoms)
    current_bonds = adsorbate_bond_set(ads_atoms)
    if template_bonds != current_bonds:
        missing = sorted(template_bonds - current_bonds)
        extra = sorted(current_bonds - template_bonds)
        return False, f"bond_changed(missing={missing}, extra={extra})"
    return True, "ok"

def adsorbate_height_from_site(atoms, sp_id, site_pos, surface_normal, slab_atom_count):
    normal = np.array(surface_normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    if len(atoms) <= slab_atom_count:
        return None
    ads_positions = atoms.get_positions()[slab_atom_count:]
    rel = ads_positions - np.array(site_pos, dtype=float)
    proj = np.dot(rel, normal)
    return float(np.min(proj))

def is_adsorbate_close_to_surface(atoms, sp_id, site_pos, surface_normal, slab_atom_count, max_height=2.2):
    height = adsorbate_height_from_site(atoms, sp_id, site_pos, surface_normal, slab_atom_count)
    if height is None:
        return False, None
    return bool(height <= max_height), height

def get_rotating_group_indices(stru, moving_idx, pivot_idx):
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

def orient_broken_bond_torsion(
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

    surface_normal = surface_normal

    normal = np.array(surface_normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return False, None, 0.0
    normal_unit = normal / normal_norm

    ads_set = set(ads_indices or [])
    bonded_pairs = adsorbate_bond_set(stru)

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

        rotating_group = get_rotating_group_indices(stru, moving_idx, pivot_idx)
        if not rotating_group:
            continue
        # Avoid rotating the anchored adsorption atoms with the moving fragment.
        if any(idx in ads_set for idx in rotating_group):
            continue

        axis_end = find_heavy_neighbor(stru, pivot_idx, exclude_idx=moving_idx)
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

def add_reactive_endpoint_site_anchor(
    stru,
    surface_normal,
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
    normal = np.array(surface_normal, dtype=float)
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

def get_imag_mode_report(ads, vib, rec_bond_global, slab_atom_count, surface_normal):
    """Analyze imaginary modes and compare bond-breaking vs adsorbate-drift character."""
    freqs = np.array(vib.get_frequencies(), dtype=complex)
    imag_indices = [
        i for i, f in enumerate(freqs)
        if (np.iscomplexobj(f) and abs(np.imag(f)) > 1e-12) or (np.isrealobj(f) and np.real(f) < 0.0)
    ]
    if len(imag_indices) == 0:
        return None

    normal = np.array(surface_normal, dtype=float)
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

def displace_and_relax_imag_mode(
    atoms,
    mode,
    sign,
    out_xyz,
    rec_bond_global,
    site_pos,
    surface_normal,
    imag_mode_displacement=0.15,
    imag_mode_relax_steps=40,
    fmax=0.10,
):
    """Displace structure along imaginary mode and perform short relaxation."""
    mode_norm = np.linalg.norm(mode)
    if mode_norm < 1e-14:
        return None

    disp_atoms = atoms.copy()
    scale = sign * imag_mode_displacement
    disp_atoms.set_positions(disp_atoms.get_positions() + scale * (mode / mode_norm))

    write(out_xyz.replace(".xyz", "_init.xyz"), disp_atoms)

    disp_atoms.calc = DP(model=MODEL)
    disp_atoms.set_constraint(build_constraints(disp_atoms, with_internal_bonds=False))

    try:
        opt = MDMin(disp_atoms, dt=0.05)
        opt.run(fmax=fmax, steps=imag_mode_relax_steps)
    except Exception:
        pass

    write(out_xyz, disp_atoms)

    i, j = rec_bond_global
    _, distances = get_distances(
        disp_atoms.get_positions(), None, disp_atoms.get_cell(), [True, True, True]
    )
    d_broken = float(distances[i, j])

    normal = np.array(surface_normal, dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    ads_positions = disp_atoms.get_positions()[len(self.slab.stru):]
    ads_center = np.mean(ads_positions, axis=0)
    height = float(np.dot(ads_center - np.array(site_pos, dtype=float), normal))

    return {
        "broken_bond": d_broken,
        "ads_height": height,
        "energy": float(disp_atoms.get_potential_energy()),
    }


def ts_bond_upper_bound(d_ref, atom_i_number, atom_j_number,
                        scale_ref=1.9, scale_cov=1.6, additive_cap=1.2):
    d_ref = float(d_ref)
    r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
    base_limit = max(scale_ref * d_ref, scale_cov * r_sum)
    additive_cap_val = d_ref + additive_cap
    return min(base_limit, additive_cap_val)

def is_ts_bond_overstretched(d_now, d_ref, atom_i_number, atom_j_number, **kwargs):
    d_max = ts_bond_upper_bound(d_ref, atom_i_number, atom_j_number, **kwargs)
    return bool(float(d_now) > float(d_max)), d_max

def ts_bond_dissociation_threshold(d_ref, atom_i_number, atom_j_number,
                                   scale_ref=1.35, scale_cov=1.25, additive=0.35):
    d_ref = float(d_ref)
    r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
    return min(scale_ref * d_ref, scale_cov * r_sum, d_ref + additive)

def imag_frequency_magnitude(freq):
    if np.iscomplexobj(freq):
        return abs(float(np.imag(freq)))
    val = float(np.real(freq))
    if val < 0.0:
        return abs(val)
    return 0.0

def significant_imag_frequencies(freqs, cutoff=20.0):
    return [f for f in freqs if imag_frequency_magnitude(f) >= cutoff]

def get_vib_indices(atoms, slab_atom_count, normal_axis, surface_layer_tol=0.6):
    n_atoms = len(atoms)
    if n_atoms == 0:
        return []
    slab_total = slab_atom_count
    slab_atom_count = max(0, min(int(slab_atom_count), n_atoms))
    axis_id = {"x": 0, "y": 1, "z": 2}[normal_axis]
    positions = atoms.get_positions()
    if slab_atom_count <= 0:
        return list(range(n_atoms))
    slab_positions = positions[:slab_atom_count]
    top_coord = float(np.max(slab_positions[:, axis_id]))
    cutoff = top_coord - float(surface_layer_tol)
    surface_indices = [idx for idx in range(slab_atom_count) if float(slab_positions[idx, axis_id]) >= cutoff]
    if n_atoms <= slab_atom_count:
        return surface_indices
    adsorbate_indices = list(range(slab_atom_count, n_atoms))
    from utils import config as cfgmod
    return cfgmod._dedupe_keep_order(adsorbate_indices + surface_indices)

def thermo_analysis(atoms, T, name="vib", indices=None, delta=0.01, nfree=2, normal_axis=None, slab_atom_count=None):
    # 移植原 _thermo_analysis，需要处理缓存等
    pass

def gas_ideal_thermo_corrections(atoms, temperature_K, pressure_pa, name, indices=None, delta=0.01, nfree=2):
    # 移植 _gas_ideal_thermo_corrections
    pass

def infer_gas_geometry(atoms):
    if len(atoms) == 1:
        return "monatomic"
    moments = np.array(atoms.get_moments_of_inertia(), dtype=float)
    if len(moments) == 0:
        return "nonlinear"
    if float(np.min(moments)) < 1e-3:
        return "linear"
    return "nonlinear"

def infer_gas_symmetry_number(atoms):
    formula = atoms.get_chemical_formula(mode="hill")
    if formula in {"H2", "N2", "O2", "F2", "Cl2", "Br2", "I2"}:
        return 2
    if formula in {"CO2", "H2O"}:
        return 2
    return 1

def energy_from_xyz(xyz_path, model_path, normal_axis, bottom_freeze_threshold):
    atoms = read(xyz_path)
    atoms.calc = DP(model=model_path)
    constraints = build_constraints(atoms, normal_axis, bottom_freeze_threshold, with_internal_bonds=False)
    atoms.set_constraint(constraints)
    return float(atoms.get_potential_energy())

# 其他需要的辅助函数：如 _enumerate_azimuth_orientation, _build_site_and_center_anchors 等
# 也可放在这里，或保留在各自 handler 中作为私有方法