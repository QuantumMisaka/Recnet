import numpy as np
from scipy.optimize import minimize
from .utils import EVectorProvider
from ase.geometry import find_mic

class CCQNUphill:
    def __init__(self):
        self.e_vector_provider = EVectorProvider()

    @staticmethod
    def _reactive_distance_from_flat(flat_coords, bond, cell, pbc):
        if bond is None:
            return None
        i, j = bond
        coords = flat_coords.reshape(-1, 3)
        if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
            return None
        rij = coords[j] - coords[i]
        rij_mic, _ = find_mic(rij[None, :], cell, pbc)
        return float(np.linalg.norm(rij_mic[0]))

    def step(self, env):
        e_vec = self.e_vector_provider.evec_ic(env.atoms, env.reactive_bonds, ic_mode='democratic')

        s_pos = self.cone_trust_region_step(
            env.g_k,
            env.B,
            env.trust_radius_uphill,
            e_vec,
            env.cos_phi
        )

        if not env.reactive_bonds:
            return s_pos

        bond = env.reactive_bonds[0]
        d0 = self._reactive_distance_from_flat(env.x_k, bond, env.atoms.get_cell(), env.atoms.get_pbc())
        d_pos = self._reactive_distance_from_flat(env.x_k + s_pos, bond, env.atoms.get_cell(), env.atoms.get_pbc())

        if d0 is None or d_pos is None:
            return s_pos

        if d_pos >= d0:
            return s_pos

        s_neg = self.cone_trust_region_step(
            env.g_k,
            env.B,
            env.trust_radius_uphill,
            -e_vec,
            env.cos_phi
        )
        d_neg = self._reactive_distance_from_flat(env.x_k + s_neg, bond, env.atoms.get_cell(), env.atoms.get_pbc())

        if d_neg is not None and d_neg > d_pos:
            if getattr(env, 'logfile', None):
                env.logfile.write(
                    f"  Uphill note: flip e-vector sign to avoid bond shrinking (d0={d0:.4f} Å, d+={d_pos:.4f} Å, d-={d_neg:.4f} Å).\n"
                )
            return s_neg

        return s_pos

    def cone_trust_region_step(self, g, B, delta, e_vec, cos_phi):
        # Placeholder for the actual implementation of the cone trust region step
        s_0 = e_vec * delta

        def target(s):
            return g.T @ s + 0.5 * s.T @ B @ s
        def jac_target(s):
            return g + B @ s
        
        def trust_constraint(s):
            return s.T @ s - delta ** 2
        def jac_trust_constraint(s):
            return 2 * s
        
        def cone_constraint(s):
            return e_vec.T @ s - cos_phi * delta
        def jac_cone_constraint(s):
            return e_vec
        
        constraints = [
            {'type': 'eq', 'fun': trust_constraint, 'jac': jac_trust_constraint},
            {'type': 'ineq', 'fun': cone_constraint, 'jac': jac_cone_constraint}
        ]

        res = minimize(target, s_0, jac=jac_target, constraints=constraints, 
                       method='SLSQP', 
                       options={'maxiter': 1000, 'ftol': 1e-6})
        if res.success:
            return res.x
        else:
            return s_0  # Fallback to initial guess if optimization fails