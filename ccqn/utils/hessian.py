import numpy as np
from scipy.linalg import eigh
from ase import Atoms

class HessianUpdater:
    def __init__(self, atoms: Atoms, model):
        self.atoms = atoms
        self.model = model
        self.hessian_approx = self._initialize_hessian()

    def _initialize_hessian(self):
        hessian_approx = []
        N = len(self.atoms) * 3
        if self.model == 'calc':
            hessian_approx = self.atoms.calc.get_hessian(self.atoms).reshape(N, N)
        else:
            hessian_approx = np.eye(N) * 70.0
        return hessian_approx
    
    def get_hessian(self):
        return self.hessian_approx

    def update(self, B, s, y, logfile=None, eigvals=None, eigvecs=None):
        # Extra parameters are accepted for compatibility with Optimizer workflow.
        self.hessian_approx = self.ts_bfgs_update(B, s, y)
        return self.hessian_approx

    def ts_bfgs_update(self, B, s, y):
        """Two-sided BFGS update of the Hessian approximation."""

        try:
            w, V = eigh(B)
            B_tilde = V @ np.diag(np.abs(w)) @ V.T
        except np.linalg.LinAlgError:
            B_tilde = B

        M = np.outer(y, y) + (s @ (B_tilde @ s)) * B_tilde
        j = y - B @ s

        denom = float(s @ (M @ s))
        if denom <= 1e-12:
            return B  # 退化情形，跳过更新
        
        u = (M @ s) / denom
        deltaB = np.outer(j, u) + np.outer(u, j) - (j.T @ s) * np.outer(u, u)

        return B + deltaB