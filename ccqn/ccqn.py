import numpy as np
from numpy.linalg import norm, eig, solve, eigh
from ase.optimize.optimize import Optimizer

from .saddle import PRFOPhase, _PRFOSolver, SellaPhase
from .uphill import CCQNUphill
from .utils import HessianUpdater, StepContext, _TrustRegionManager

class CCQN(Optimizer):
    def __init__(
        self,
        atoms,
        restart=None,
        logfile='-',
        trajectory=None,
        master=None,
        e_vector_method='ic',
        product_atoms=None,
        reactive_bonds=None,
        ic_mode='democratic',
        cos_phi=0.5,
        trust_radius_uphill=0.1,
        trust_radius_saddle_initial=0.05,
        trust_radius_saddle_min=5e-3,
        trust_radius_saddle_max=0.2,
        idpp_images=7,
        use_idpp=False,
        hessian_model='ts-bfgs',
        saddle_optimizer='prfo',
        sella_kwargs=None,
    ):
        super().__init__(atoms, restart=restart, logfile=logfile, trajectory=trajectory, master=master)

        self.e_vector_method = e_vector_method
        self.ic_mode = ic_mode.lower()
        self.product_atoms = product_atoms
        self.reactive_bonds = reactive_bonds or []
        self.idpp_images = idpp_images
        self.use_idpp = use_idpp

        self.cos_phi = cos_phi
        self.trust_radius_uphill = trust_radius_uphill
        self.trust_radius_saddle = trust_radius_saddle_initial
        self.trust_radius_saddle_min = trust_radius_saddle_min
        self.trust_radius_saddle_max = trust_radius_saddle_max
        self.trust_radius_reset_value = trust_radius_saddle_initial
        self.saddle_optimizer = str(saddle_optimizer).lower()
        if self.saddle_optimizer not in {'prfo', 'sella'}:
            raise ValueError("saddle_optimizer must be either 'prfo' or 'sella'.")
        self.sella_kwargs = dict(sella_kwargs or {})

        self._hessian_mgr = HessianUpdater(self.atoms, hessian_model)
        self.B = self._hessian_mgr.get_hessian()

        # Components
        self._mode_selector = self  # reuse local select implementation
        self._prfo_solver = _PRFOSolver()
        self._trust_mgr = _TrustRegionManager(
            self.trust_radius_saddle, self.trust_radius_saddle_min, self.trust_radius_saddle_max
        )
        self._prfo_phase = PRFOPhase(self._prfo_solver, self._trust_mgr)
        self._sella_phase = None
        if self.saddle_optimizer == 'sella':
            self._sella_phase = SellaPhase(
                self.atoms,
                trust_radius_saddle_initial=self.trust_radius_saddle,
                logfile=self.logfile,
                trajectory=trajectory,
                master=master,
                sella_kwargs=self.sella_kwargs,
            )
        self._uphill_phase = CCQNUphill()

        # Optimizer state
        self.mode = 'uphill'
        self.g_k_minus_1 = None
        self.pos_k_minus_1 = None
        self.energy_k_minus_1 = None
        self.eigvals = None
        self.eigvecs = None
    
    def check(self):
        pass

    def select(self, current_mode, eigvals, logfile=None):
        """
        Determine the next mode based on Hessian eigenvalues.
        
        Returns:
            new_mode (str): The next mode ('uphill' or 'prfo').
            trust_reset (float or None): If not None, the trust radius should be reset to this value.
            reason (str): The reason for the mode switch (or None if no switch).
        """
        # eigvals is assumed to be sorted (ascending)
        min_eig = eigvals[0]
        if logfile:
            logfile.write(f"\nThe smallest eigenvalue of Hessian is {min_eig:.4e}")
            
        new_mode = current_mode
        trust_reset = None
        reason = None

        if current_mode == 'uphill':
            if min_eig < -1e-6:
                new_mode = 'prfo'
                reason = f"Min eigenvalue ({min_eig:.4e}) < -1e-6"
                if logfile:
                    logfile.write(f"\nSwitching to 'prfo' mode. Reason: {reason}")
                trust_reset = self.trust_radius_reset_value
        elif current_mode == 'prfo':
            if min_eig > 1e-2:
                new_mode = 'uphill'
                reason = f"Min eigenvalue ({min_eig:.4e}) > 1e-2"
                if logfile:
                    logfile.write(f"\nSwitching to 'uphill' mode. Reason: {reason}")
        
        return new_mode, trust_reset, reason

    # def converged(self, atoms, forces, fmax, mode):
    #     if forces is None:
    #         forces = atoms.get_forces()
    #     elif hasattr(forces, 'ndim') and forces.ndim == 1:
    #         forces = forces.reshape(-1, 3)
    #     f = np.sqrt((forces ** 2).sum(axis=1).max())
    #     return (f < fmax) and (mode == 'prfo')
    
    def converged(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        elif hasattr(forces, 'ndim') and forces.ndim == 1:
            forces = forces.reshape(-1, 3)
        f = np.sqrt((forces ** 2).sum(axis=1).max())
        return (f < self.fmax) and (self.mode == 'prfo')

    def _reactive_bond_distance(self, flat_positions):
        if not self.reactive_bonds:
            return None
        try:
            i, j = self.reactive_bonds[0]
            coords = flat_positions.reshape(-1, 3)
            if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
                return None
            return float(np.linalg.norm(coords[i] - coords[j]))
        except Exception:
            return None

    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()
        g_k = -f.flatten()
        x_k = self.atoms.get_positions().flatten()
        e_k = self.atoms.get_potential_energy()
        if self.nsteps > 0:
            s_k_prev = x_k - self.pos_k_minus_1
            y_k_prev = g_k - self.g_k_minus_1
            if np.linalg.norm(s_k_prev) > 1e-7:
                self.B = self._hessian_mgr.update(self.B, s_k_prev, y_k_prev, self.logfile, self.eigvals, self.eigvecs)
        try:
            # Diagonalize first (consistent with new architecture)
            eigvals, eigvecs = eigh(self.B)
            
            # Select mode using new signature
            new_mode, trust_reset, reason = self._mode_selector.select(self.mode, eigvals, self.logfile)
            
            self.mode = new_mode
            if trust_reset is not None:
                self.trust_radius_saddle = trust_reset
                # Also update manager state
                if hasattr(self._trust_mgr, 'set_radius'):
                    self._trust_mgr.set_radius(trust_reset)
        except Exception as e:
            self.logfile.write(f"Hessian diagonalization or mode selection failed ({e}). Resetting Hessian.\n")
            self.B = self._initialize_hessian()
            eigvals, eigvecs = eigh(self.B)
            self.mode = 'uphill'
        
        # Store for next step's BFGS update
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        fmax = np.sqrt((f**2).sum(axis=1).max())
        self.logfile.write(
            f"Step {self.nsteps:3d}: Mode='{self.mode}', SaddleOpt='{self.saddle_optimizer}', "
            f"E={e_k:.4f}, Fmax={fmax:.4f}, Trust(Saddle)={self.trust_radius_saddle:.4e}\n"
        )
        ctx = StepContext(self.atoms, self.B, g_k, x_k, e_k, eigvals, eigvecs, self.trust_radius_uphill, self.trust_radius_saddle, self.cos_phi, self.e_vector_method, getattr(self, "product_atoms", None), getattr(self, "idpp_images", 7), getattr(self, "use_idpp", False), getattr(self, "reactive_bonds", None), self.ic_mode, self.pos_k_minus_1, self.g_k_minus_1, self.energy_k_minus_1, self.logfile)
        if self.mode == 'uphill':
            s_k = self._uphill_phase.step(ctx)
        else:
            if self.saddle_optimizer == 'sella':
                try:
                    s_k, self.trust_radius_saddle = self._sella_phase.run(ctx)
                except Exception as exc:
                    fallback_radius = getattr(exc, 'suggested_radius', None)
                    if fallback_radius is not None and np.isfinite(fallback_radius) and fallback_radius > 0.0:
                        self.trust_radius_saddle = float(fallback_radius)
                    else:
                        self.trust_radius_saddle = max(self.trust_radius_saddle * 0.5, 1e-3)
                    if hasattr(self._trust_mgr, 'set_radius'):
                        self._trust_mgr.set_radius(self.trust_radius_saddle)
                    self.logfile.write(
                        f"Sella step failed ({exc}); fallback to PRFO with trust radius {self.trust_radius_saddle:.4e}.\n"
                    )
                    s_k, self.trust_radius_saddle = self._prfo_phase.run(ctx)
            else:
                s_k, self.trust_radius_saddle = self._prfo_phase.run(ctx)
        x_next = x_k + s_k
        self.atoms.set_positions(x_next.reshape(-1, 3))

        step_norm = float(np.linalg.norm(s_k))
        bond_dist = self._reactive_bond_distance(x_next)
        if bond_dist is None:
            self.logfile.write(f"  StepInfo: |dR|={step_norm:.4e}\n")
        else:
            self.logfile.write(f"  StepInfo: |dR|={step_norm:.4e}, d(reactive)={bond_dist:.4f} Å\n")

        self.pos_k_minus_1 = x_k
        self.g_k_minus_1 = g_k
        self.energy_k_minus_1 = e_k
