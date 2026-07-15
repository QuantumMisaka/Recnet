class Environment:
    def __init__(self, atoms, B, g_k, x_k, e_k, trust_radius_uphill, cos_phi, reactive_bonds):
        self.atoms = atoms
        self.B = B
        self.g_k = g_k
        self.x_k = x_k
        self.e_k = e_k
        # self.eigvals = eigvals
        # self.eigvecs = eigvecs
        self.trust_radius_uphill = trust_radius_uphill
        self.cos_phi = cos_phi
        # self.e_vector_method = e_vector_method
        self.reactive_bonds = reactive_bonds
        # self.ic_mode = ic_mode
