import numpy as np

class Material:
    def __init__(self, E, nu, plane_stress=False):
        """Initialize material properties with validation"""
        # Convert and validate inputs
        self.youngs_modulus = np.array(E, dtype=np.float64)
        self.poissons_ratio = np.array(nu, dtype=np.float64)
        
        assert self.youngs_modulus.shape == self.poissons_ratio.shape, \
            "Material properties dimension mismatch"
        assert np.all(self.youngs_modulus > 0), "Young's modulus must be positive"
        assert np.all((self.poissons_ratio > 0) & (self.poissons_ratio < 0.5)), \
            "Poisson's ratio must be in (0, 0.5)"
            
        self.plane_stress = plane_stress
        self._calculate_elastic_constants()
        
    def _calculate_elastic_constants(self):
        """Compute Lame parameters based on material model"""
        if self.plane_stress:
            # For plane stress: λ = (E*ν)/(1-ν²)
            self.lambd = (self.youngs_modulus * self.poissons_ratio) / \
                        (1 - self.poissons_ratio**2)
        else:  # Plane strain
            # For plane strain: λ = (E*ν)/((1+ν)(1-2ν))
            self.lambd = (self.youngs_modulus * self.poissons_ratio) / \
                        ((1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
        
        # Shear modulus (same for both cases)
        self.mu = self.youngs_modulus / (2 * (1 + self.poissons_ratio))

    def get_constitutive_matrix(self):
        """Generate constitutive matrix for plane stress/strain"""
        n_phases = len(self.youngs_modulus)
        C = np.zeros((n_phases, 3, 3))
        
        for i in range(n_phases):
            if self.plane_stress:
                # Plane stress constitutive matrix
                factor = self.youngs_modulus[i] / (1 - self.poissons_ratio[i]**2)
                C[i, :, :] = factor * np.array([
                    [1, self.poissons_ratio[i], 0],
                    [self.poissons_ratio[i], 1, 0],
                    [0, 0, (1 - self.poissons_ratio[i])/2]
                ])
            else:
                # Plane strain constitutive matrix
                C[i, :, :] = np.array([
                    [self.lambd[i] + 2*self.mu[i], self.lambd[i], 0],
                    [self.lambd[i], self.lambd[i] + 2*self.mu[i], 0],
                    [0, 0, self.mu[i]]
                ])
        
        # 添加本构矩阵验证
        for i in range(len(self.youngs_modulus)):
            if np.any(np.isinf(C[i])) or np.any(np.isnan(C[i])):
                raise ValueError(f"Invalid constitutive matrix at phase {i}")
        return C 