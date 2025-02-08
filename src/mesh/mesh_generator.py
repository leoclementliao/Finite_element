import numpy as np
from scipy.spatial import Delaunay

class MeshGenerator:
    def __init__(self, length, nx):
        self.length = length
        self.nx = nx
        
    def generate_mesh(self):
        """Generate 2D mesh with triangular elements"""
        h = self.length
        ny = self.nx
        [Mx, My] = np.meshgrid(np.linspace(0, self.length, self.nx), 
                              np.linspace(0, h, ny))
        Nx = np.reshape(Mx.T, (self.nx * ny)) - self.length/2
        Ny = np.reshape(My.T, (self.nx * ny)) - h/2
        M_Nxy = np.array([Nx, Ny]).T
        M_tri = Delaunay(M_Nxy).simplices.copy()
        
        Ne = M_tri.shape[0]
        M_GDof = np.array([2*M_tri[:,0], 2*M_tri[:,0]+1,
                          2*M_tri[:,1], 2*M_tri[:,1]+1,
                          2*M_tri[:,2], 2*M_tri[:,2]+1]).T
        
        # Calculate element centers
        M_EleCentre = np.zeros((Ne, 2))
        for i in range(Ne):
            nodes = M_tri[i,:]
            X = np.mean(M_Nxy[nodes,0])
            Y = np.mean(M_Nxy[nodes,1])
            M_EleCentre[i,:] = np.array([X,Y])
            
        # Find boundary nodes
        MinX = min(Nx)
        MaxX = max(Nx)
        MinY = min(Ny)
        MaxY = max(Ny)
        
        BordD = np.where(abs(Ny-MinY)<1e-6)[0]
        BordR = np.where(abs(Nx-MaxX)<1e-6)[0]
        BordU = np.where(abs(Ny-MaxY)<1e-6)[0]
        BordL = np.where(abs(Nx-MinX)<1e-6)[0]
        
        # Sort boundary nodes
        BordD = BordD[Nx[BordD].argsort()][1:]
        BordR = BordR[Ny[BordR].argsort()]
        BordU = BordU[Nx[BordU].argsort()][1:]
        BordL = BordL[Ny[BordL].argsort()]
        
        return {
            'nodes': M_Nxy,
            'elements': M_tri,
            'dof_map': M_GDof,
            'element_centers': M_EleCentre,
            'boundary': {
                'bottom': BordD,
                'right': BordR,
                'top': BordU,
                'left': BordL
            }
        } 