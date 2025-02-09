import numpy as np
from scipy.sparse import coo_matrix

# Main Finite Element solver class
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
            M3_B[i,:,:] = 0.5/Ae[i] * np.array([
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
            Ke = Ae[i] * np.dot(np.dot(M_B.T, C[phase_map[i],:,:]), M_B)
            # Assemble into global matrix
            dofs = self.dof_map[i,:]
            KGlob[np.ix_(dofs, dofs)] += Ke
            
        return KGlob
        
    def apply_boundary_conditions(self, K, F, boundary_type='periodic'):
        """Apply boundary conditions with force loading"""
        # Apply small horizontal tension
        right_nodes = self.mesh['boundary']['right']
        F[2*right_nodes] = 1e3  # 1kN in X-direction
        
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
        
        # 添加约束验证
        print(f"Applying {boundary_type} boundary conditions")
        print(f"Number of constraints: {C.shape[0]}")
        print(f"Rank deficiency: {K.shape[0] - np.linalg.matrix_rank(K)}")
        
        # 添加矩阵秩检查
        print(f"Original DOF: {K.shape[0]}")
        print(f"Constraint matrix rank: {np.linalg.matrix_rank(C)}/{C.shape[0]}")
        print(f"Augmented system size: {K_aug.shape[0]}")
        
        return K_aug, F_aug
        
    def solve(self, K, F):
        """Solve system with enhanced numerical stability"""
        # Check matrix condition
        cond_number = np.linalg.cond(K)
        print(f"Condition number: {cond_number:.2e}")
        
        # Add adaptive regularization
        reg_scale = max(1e-10, 1e-12 * cond_number)
        reg = np.eye(K.shape[0]) * reg_scale
        
        # Use least squares for ill-conditioned systems
        solution = np.linalg.lstsq(K + reg, F, rcond=None)[0]
        
        # Verify solution validity
        if np.any(np.isnan(solution)):
            raise RuntimeError("Solution contains NaN values. Check boundary conditions.")
        
        return solution
        
    def _get_periodic_bc(self):
        """修正旋转约束方程"""
        bord = self.mesh['boundary']
        bottom, top = bord['bottom'], bord['top']
        left, right = bord['left'], bord['right']

        # Verify node pairing
        assert len(bottom) == len(top), "Top/bottom node count mismatch"
        assert len(left) == len(right), "Left/right node count mismatch"

        # Create constraint pairs
        constraint_pairs = []
        # Vertical pairs (bottom-top)
        for b, t in zip(bottom, top):
            constraint_pairs.append((b, t))
        # Horizontal pairs (left-right)
        for l, r in zip(left, right):
            constraint_pairs.append((l, r))

        # Build constraint matrix
        n_constraints = 2 * len(constraint_pairs)
        C = np.zeros((n_constraints + 3, self.n_dof))  # 3 rigid body constraints
        
        # Add displacement equality constraints
        for i, (n1, n2) in enumerate(constraint_pairs):
            # X-direction
            C[2*i, 2*n1] = 1
            C[2*i, 2*n2] = -1
            # Y-direction
            C[2*i+1, 2*n1+1] = 1
            C[2*i+1, 2*n2+1] = -1

        # Add anti-rotation constraints (fix center node)
        center = np.mean(self.mesh['nodes'], axis=0)
        ref_node = np.argmin(np.linalg.norm(self.mesh['nodes'] - center, axis=1))
        C[-3, 2*ref_node] = 1      # 固定X位移
        C[-2, 2*ref_node+1] = 1    # 固定Y位移
        C[-1, 2*ref_node] = -1    # X位移项
        C[-1, 2*ref_node+1] = 1    # Y位移项
        Ud = np.zeros(C.shape[0])
        Ud[-1] = 0  # u_y - u_x = 0
        
        return C, Ud

    # def apply_strain_loading(self, strain):
    #     ... 