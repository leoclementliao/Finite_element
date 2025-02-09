import numpy as np
import pytest
from ..fem_solver import FEMSolver
from ...mesh.mesh_generator import MeshGenerator
from ...mechanics.material import Material

@pytest.fixture
def simple_problem():
    """Create a simple test problem"""
    # Create mesh
    mesh_gen = MeshGenerator(length=1.0, nx=4)
    mesh = mesh_gen.generate_mesh()
    
    # Create material
    material = Material(E=[1.0, 10.0], nu=[0.3, 0.3])
    
    # Create solver
    solver = FEMSolver(mesh, material)
    
    return solver, mesh, material

def test_strain_displacement_matrix(simple_problem):
    """Test B matrix computation"""
    solver, mesh, _ = simple_problem
    B = solver.compute_strain_displacement_matrix()
    
    # Test shape
    assert B.shape[0] == solver.n_elements
    assert B.shape[1] == 3  # εxx, εyy, γxy
    assert B.shape[2] == 6  # 2 DOFs per node * 3 nodes
    
    # Test if B matrices satisfy compatibility conditions
    for i in range(solver.n_elements):
        # Sum of each row should be zero (rigid body motion)
        assert np.allclose(np.sum(B[i, 0, ::2]), 0)  # εxx
        assert np.allclose(np.sum(B[i, 1, 1::2]), 0)  # εyy

def test_stiffness_matrix(simple_problem):
    """Test global stiffness matrix assembly"""
    solver, mesh, _ = simple_problem
    
    # Create simple phase map (all elements are phase 0)
    phase_map = np.zeros(solver.n_elements, dtype=int)
    
    K = solver.assemble_stiffness_matrix(phase_map)
    
    # Test symmetry
    assert np.allclose(K, K.T)
    
    # Test positive definiteness
    eigenvals = np.linalg.eigvals(K)
    assert np.all(eigenvals[eigenvals > 1e-10] > 0)

def test_periodic_boundary_conditions(simple_problem):
    """Test periodic boundary condition generation"""
    solver, mesh, _ = simple_problem
    
    C, Ud = solver._get_periodic_bc()
    
    # Test constraint matrix dimensions
    n_constraints = 2 * (len(mesh['boundary']['bottom']) + len(mesh['boundary']['left']))
    assert C.shape == (n_constraints, solver.n_dof)
    
    # Test if displacement vector has correct size
    assert len(Ud) == n_constraints
