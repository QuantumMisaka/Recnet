import numpy as np
from numpy.linalg import norm
from ase.geometry import find_mic

class EVectorProvider:
    def evec_ic(self, atoms, reactive_bonds, ic_mode='democratic'):

        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        pos = atoms.get_positions()
        forces = atoms.get_forces()
        N = len(atoms) * 3

        bonds = np.array(reactive_bonds, dtype=int)
        i_idx, j_idx = bonds[:, 0], bonds[:, 1]
        raw_v_ij = pos[j_idx] - pos[i_idx]
        v_ij, _ = find_mic(raw_v_ij, cell, pbc)
        norm_v = norm(v_ij, axis=1)
        valid = norm_v > 1e-8
        v_ij = v_ij[valid]
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        if v_ij.shape[0] == 0:
            return np.zeros(N)
        
        f_i = forces[i_idx]
        f_j = forces[j_idx]

        v_f_j = np.sum(v_ij * f_j, axis=1)
        v_f_i = np.sum(v_ij * f_i, axis=1)
        denom = np.sum(v_ij * v_ij, axis=1)
        p_ij = v_ij * (v_f_j / denom)[:, None] - v_ij * (v_f_i / denom)[:, None]

        E = np.zeros_like(pos)
        if ic_mode == 'democratic':
            norm_p = norm(p_ij, axis=1)
            valid = norm_p > 1e-8
            if np.sum(valid) == 0:
                return np.zeros(N)
            p_ij = p_ij[valid] / norm_p[valid][:, None]
            np.add.at(E, i_idx[valid], p_ij)
            np.add.at(E, j_idx[valid], -p_ij)
        else:
            np.add.at(E, i_idx, p_ij)
            np.add.at(E, j_idx, -p_ij)

        e_vec = E.flatten()
        n = norm(e_vec)
        if n < 1e-8:
            return np.zeros_like(e_vec)
        return e_vec / n

