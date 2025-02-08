import numpy as np
from scipy.sparse import coo_matrix

class FEMSolver:
    def __init__(self, mesh, material):
        """
        Initialize FEM solver
        
        Args:
            mesh: Dictionary containing mesh information
            material: Material object containing material properties
        """
        self.mesh = mesh
        self.material = material
        self.nodes = mesh['nodes']
        self.elements = mesh['elements']
        self.dof_map = mesh['dof_map']
        self.n_elements = len(self.elements)
        self.n_nodes = len(self.nodes)
        self.n_dof = 2 * self.n_nodes
        
    def compute_strain_displacement_matrix(self):
        """Compute B matrix for each element"""
        Ne = self.n_elements
        Ae = self.mesh['element_area']
        M3_B = np.zeros((Ne, 3, 6))
        
        for i in range(Ne):
            nodes = self.elements[i,:]
            x = self.nodes[nodes,0]
            y = self.nodes[nodes,1]
            M3_B[i,:,:] = 0.5/Ae * np.array([
                [y[1]-y[2], 0, y[2]-y[0], 0, y[0]-y[1], 0],
                [0, x[2]-x[1], 0, x[0]-x[2], 0, x[1]-x[0]],
                [x[2]-x[1], y[1]-y[2], x[0]-x[2], y[2]-y[0], x[1]-x[0], y[0]-y[1]]
            ])
        return M3_B 
    
    def assemble_stiffness_matrix(self, phase_map):
        """
        Assemble global stiffness matrix
        
        Args:
            phase_map: Array indicating material phase for each element
        """
        Ne = self.n_elements
        Ae = self.mesh['element_area']
        M3_B = self.compute_strain_displacement_matrix()
        C = self.material.get_constitutive_matrix()
        
        # Initialize global stiffness matrix
        KGlob = np.zeros((self.n_dof, self.n_dof))
        
        for i in range(Ne):
            M_B = M3_B[i,:,:]
            # Get element stiffness matrix
            Ke = Ae * np.dot(np.dot(M_B.T, C[phase_map[i],:,:]), M_B)
            # Assemble into global matrix
            dofs = self.dof_map[i,:]
            KGlob[np.ix_(dofs, dofs)] += Ke
            
        return KGlob
        
    def apply_boundary_conditions(self, K, F, boundary_type='periodic'):
        """
        Apply boundary conditions to system
        
        Args:
            K: Global stiffness matrix
            F: Global force vector
            boundary_type: Type of boundary conditions ('periodic' or 'KUBC')
        """
        if boundary_type == 'periodic':
            C, Ud = self._get_periodic_bc()
        else:  # KUBC
            C, Ud = self._get_kubc()
            
        # Augmented system with Lagrange multipliers
        n_constraints = len(Ud)
        K_aug = np.block([
            [K, C.T],
            [C, np.zeros((n_constraints, n_constraints))]
        ])
        F_aug = np.hstack([F, Ud])
        
        return K_aug, F_aug
        
    def solve(self, K, F):
        """
        Solve the system KU = F
        
        Args:
            K: System matrix
            F: Force vector
            
        Returns:
            U: Displacement vector
        """
        return np.linalg.solve(K, F)
        
    def _get_periodic_bc(self):
        """Generate periodic boundary conditions"""
        # Get boundary nodes
        bord = self.mesh['boundary']
        BordD, BordU = bord['bottom'], bord['top']
        BordL, BordR = bord['left'], bord['right']
        
        # Initialize constraint matrix and displacement vector
        n_constraints = 2 * (len(BordD) + len(BordL))
        C = np.zeros((n_constraints, self.n_dof))
        Ud = np.zeros(n_constraints)
        
        # Add constraints for vertical boundaries
        cnt = 0
        for i in range(len(BordD)):
            Ba, Bc = BordD[i], BordU[i]
            C[2*cnt:2*cnt+2, 2*np.array([Ba,Bc])] = np.array([[-1,1], [-1,1]])
            C[2*cnt:2*cnt+2, 2*np.array([Ba,Bc])+1] = np.array([[-1,1], [-1,1]])
            cnt += 1
            
        # Add constraints for horizontal boundaries
        for i in range(len(BordL)):
            Ba, Bc = BordL[i], BordR[i]
            C[2*cnt:2*cnt+2, 2*np.array([Ba,Bc])] = np.array([[-1,1], [-1,1]])
            C[2*cnt:2*cnt+2, 2*np.array([Ba,Bc])+1] = np.array([[-1,1], [-1,1]])
            cnt += 1
            
        return C, Ud 