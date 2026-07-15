import numpy as np


def get_bond_connections(atoms, shift=0, cutoff=1.2, bond_type="nosurf"):
    from ase.data import covalent_radii
    from ase.geometry import get_distances

    if bond_type == "nosurf":
        atoms = atoms[shift:]

    _, distances = get_distances(atoms.get_positions(), None, atoms.get_cell(), [True, True, True])
    radii = [covalent_radii[atom.number] for atom in atoms]
    threshold_matrix = np.add.outer(radii, radii) * cutoff
    bond_matrix = distances < threshold_matrix

    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if bond_matrix[i, j]:
                bonds.append((i, j))

    if bond_type == "nosurf":
        return bonds

    clear_bonds = []
    for bond in bonds:
        if bond[0] < shift and bond[1] < shift:
            continue

        def get_idx(idx):
            return "X" if idx < shift else idx - shift

        bond_reidx = (get_idx(bond[0]), get_idx(bond[1]))
        if bond_reidx not in clear_bonds:
            clear_bonds.append(bond_reidx)

    return clear_bonds


def rotate_about_ads_vertical(stru, ads_idx, angle_deg, surface_normal):
    """
    绕吸附原子并沿表面法向的轴旋转结构。
    """
    axis = np.array(surface_normal, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        raise ValueError("surface_normal 长度为零，无法定义旋转轴")
    axis /= norm

    if len(ads_idx) == 1:
        center = stru.get_positions()[ads_idx[0]]
    elif len(ads_idx) > 1:
        center = np.mean(stru.get_positions()[np.array(ads_idx)], axis=0)
    elif len(ads_idx) == 0:
        center = np.mean(stru.get_positions(), axis=0)
    else:
        raise ValueError("ads_idx 长度无效，无法确定旋转中心")

    stru.rotate(a=angle_deg, v=axis, center=center, rotate_cell=False)


def rotate_about_ads_inner_angle(
    stru,
    rec_bond,
    ads_indices,
    surface_normal,
    target_min_angle_deg=50.0,
):
    """
    以吸附原子为中心，将断键方向旋转到与表面法向至少形成指定内角。

    返回
    -------
    tuple(float, float)
        (旋转前内角, 实际旋转角)
    """
    def _inner_angle_deg(vec, normal_unit):
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-12:
            return 0.0
        vec_unit = vec / vec_norm
        cos_theta = np.dot(vec_unit, normal_unit)
        theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return 180.0 - theta if theta > 90.0 else theta

    def _rotate_vec(vec, axis_unit, angle_deg):
        theta = np.deg2rad(angle_deg)
        return (
            vec * np.cos(theta)
            + np.cross(axis_unit, vec) * np.sin(theta)
            + axis_unit * np.dot(axis_unit, vec) * (1.0 - np.cos(theta))
        )

    pos = stru.get_positions()
    pos1 = pos[rec_bond[0]]
    pos2 = pos[rec_bond[1]]
    bond_vec = pos2 - pos1

    bond_norm = np.linalg.norm(bond_vec)
    if bond_norm < 1e-8:
        return 0.0, 0.0

    normal = np.array(surface_normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        raise ValueError("surface_normal 长度为零，无法定义旋转")

    normal_unit = normal / normal_norm
    bond_unit = bond_vec / bond_norm
    angle = _inner_angle_deg(bond_unit, normal_unit)

    rotate_angle = target_min_angle_deg - angle
    if rotate_angle <= 0.0:
        return angle, 0.0

    rot_axis = np.cross(normal_unit, bond_unit)
    axis_norm = np.linalg.norm(rot_axis)
    if axis_norm < 1e-8:
        fallback_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback_axis, normal_unit)) > 0.9:
            fallback_axis = np.array([0.0, 1.0, 0.0])
        rot_axis = np.cross(normal_unit, fallback_axis)
        axis_norm = np.linalg.norm(rot_axis)
        if axis_norm < 1e-8:
            return angle, 0.0

    rot_axis /= axis_norm

    angle_plus = _inner_angle_deg(_rotate_vec(bond_unit, rot_axis, rotate_angle), normal_unit)
    angle_minus = _inner_angle_deg(_rotate_vec(bond_unit, rot_axis, -rotate_angle), normal_unit)
    if angle_minus > angle_plus:
        rot_axis = -rot_axis

    if ads_indices:
        if rec_bond[0] in ads_indices:
            ads_idx = rec_bond[0]
        elif rec_bond[1] in ads_indices:
            ads_idx = rec_bond[1]
        else:
            ads_idx = ads_indices[0]
    else:
        ads_idx = rec_bond[0]

    center = pos[ads_idx]
    stru.rotate(a=rotate_angle, v=rot_axis, center=center, rotate_cell=False)
    return angle, rotate_angle


def rotate_atom_around_axis(point, axis_point, axis_unit, angle_deg):
    """Rotate one point around an axis using Rodrigues formula."""
    theta = np.deg2rad(angle_deg)
    rel = point - axis_point
    return (
        axis_point
        + rel * np.cos(theta)
        + np.cross(axis_unit, rel) * np.sin(theta)
        + axis_unit * np.dot(axis_unit, rel) * (1.0 - np.cos(theta))
    )
