import numpy as np

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