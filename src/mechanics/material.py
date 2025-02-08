import numpy as np

class Material:
    def __init__(self, young_modulus, poisson_ratio):
        """
        Initialize material properties
        
        Args:
            young_modulus: Young's modulus (E) for different phases
            poisson_ratio: Poisson's ratio (nu) for different phases
        """
        self.E = np.array(young_modulus)
        self.nu = np.array(poisson_ratio)
        self.mu = 0.5 * self.E / (1 + self.nu)
        self.lambd = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        
    def get_constitutive_matrix(self):
        """Generate constitutive matrix for each phase"""
        n_phases = len(self.E)
        C = np.zeros((n_phases, 3, 3))
        
        for i in range(n_phases):
            lambda_i = self.lambd[i]
            mu_i = self.mu[i]
            C[i,:,:] = np.array([
                [lambda_i + 2*mu_i, lambda_i, 0],
                [lambda_i, lambda_i + 2*mu_i, 0],
                [0, 0, mu_i]
            ])
            
        return C 