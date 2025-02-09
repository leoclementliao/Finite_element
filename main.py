#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from src.mechanics.material import Material
from src.mesh.mesh_generator import MeshGenerator
from src.solver.fem_solver import FEMSolver

# Configuration parameters
CONFIG = {
    "length": 1.0,         # length of the domain
    "mesh_resolution": 60, # number of elements along the length
    "inclusions": 2,       # number of inclusions
    "radius": 0.2,         # radius of the inclusion
    "material_props": {
        "E": [1e9, 50e9],  # elastic modulus
        "nu": [0.25, 0.3]  # poisson's ratio
    },
    "applied_force": 5e4,  # uniform force loading
    "plane_stress": True
}

def classify_phases(mesh_data, num_inclusions, radius):
    """Identify elements containing inclusions"""
    centers = generate_inclusion_centers(mesh_data, num_inclusions, radius)
    element_centers = mesh_data['element_centers']
    phase_map = np.zeros(len(element_centers), dtype=int)
    
    # Mark inclusion elements
    for center in centers:
        distances = np.linalg.norm(element_centers - center, axis=1)
        phase_map[distances < radius] = 1
        
    return phase_map

def generate_inclusion_centers(mesh_data, num_inclusions, radius):
    """Generate valid inclusion centers with spacing check"""
    L = np.max(mesh_data['nodes'])
    centers = []
    
    for _ in range(num_inclusions):
        while True:
            new_center = np.random.rand(2) * L - L/2
            if all(np.linalg.norm(new_center - c) > 2.2*radius for c in centers):
                centers.append(new_center)
                break
                
    return np.array(centers)

def main():
    # Initialize components
    mesh_gen = MeshGenerator(CONFIG['length'], CONFIG['mesh_resolution'])
    mesh_data = mesh_gen.generate_mesh()
    material = Material(
        E=CONFIG['material_props']['E'],
        nu=CONFIG['material_props']['nu'],
        plane_stress=CONFIG['plane_stress']
    )
    solver = FEMSolver(mesh_data, material)

    # Phase identification
    phase_map = classify_phases(mesh_data, 
                              CONFIG['inclusions'], 
                              CONFIG['radius'])

    # Apply tension load
    right_boundary = mesh_data['boundary']['right']
    left_boundary = mesh_data['boundary']['left']
    F = np.zeros(solver.n_dof)
    total_force = CONFIG['applied_force']
    force_per_node = total_force / len(right_boundary)
    F[2*left_boundary] = -force_per_node  # 左侧反向力
    F[2*right_boundary] = force_per_node   # 右侧正向力

    # Assemble and solve system
    K = solver.assemble_stiffness_matrix(phase_map)
    K_aug, F_aug = solver.apply_boundary_conditions(K, F, 'periodic')
    U = solver.solve(K_aug, F_aug)[:solver.n_dof]

    # Post-processing
    displacement = U.reshape(-1, 2)
    magnitude = np.linalg.norm(displacement, axis=1)
    
    print(f"Displacement statistics:")
    print(f"Max: {np.max(magnitude):.2e} m")
    print(f"Min: {np.min(magnitude):.2e} m")
    print(f"Mean: {np.mean(magnitude):.2e} m")

    # 在施加载荷后添加检查
    print(f"Total applied force: {np.sum(F[::2]):.2f} N (X-direction)")
    print(f"Total applied force: {np.sum(F[1::2]):.2f} N (Y-direction)")

    # Visualization
    visualize_results(mesh_data, phase_map, U, left_boundary)

    # Check mesh quality
    validate_mesh(mesh_data)

def visualize_results(mesh_data, phase_map, U, left_boundary):
    """Visualize results with boundary markers"""
    displacement = U.reshape(-1, 2)
    magnitude = np.linalg.norm(displacement, axis=1)
    
    # Check for invalid displacement values
    if np.any(np.isnan(magnitude)):
        raise ValueError("Displacement contains NaN values. Cannot visualize.")
    
    # Calculate automatic scaling factor
    max_disp = np.nanmax(magnitude)
    scale = 50 / max_disp if max_disp > 1e-6 else 1.0
    
    plt.figure(figsize=(18, 6))
    
    # Phase distribution
    plt.subplot(131)
    plot_phase_distribution(mesh_data, phase_map)
    
    # Displacement contour
    plt.subplot(132)
    plot_displacement_contour(mesh_data, magnitude)
    
    # Deformed mesh
    plt.subplot(133)
    plot_deformed_mesh(mesh_data, displacement, magnitude)
    
    # Add constraint node marker
    if len(left_boundary) > 0:
        ref_node = left_boundary[0]  # 使用原始节点索引
        plt.scatter(mesh_data['nodes'][ref_node,0], 
                   mesh_data['nodes'][ref_node,1], 
                   c='red', s=50, label='Constraint Node')
    
    plt.tight_layout()
    plt.show()

def plot_phase_distribution(mesh_data, phase_map):
    """Plot material phase distribution"""
    for phase in [0, 1]:
        mask = phase_map == phase
        plt.triplot(*mesh_data['nodes'].T, 
                   triangles=mesh_data['elements'][mask],
                   color='blue' if phase == 0 else 'red',
                   linewidth=0.5)
    plt.title("Material Phase Distribution\n(Blue: Matrix, Red: Inclusion)")
    plt.axis('equal')

def plot_displacement_contour(mesh_data, magnitude):
    """Plot displacement contour"""
    levels = np.linspace(np.min(magnitude), np.max(magnitude), 20)
    tcf = plt.tricontourf(*mesh_data['nodes'].T, magnitude, 
                         levels=levels, cmap='viridis')
    plt.colorbar(tcf, label='Displacement Magnitude [m]')
    plt.title("Displacement Field Contour")
    plt.axis('equal')

def plot_deformed_mesh(mesh_data, displacement, magnitude):
    """Plot deformed mesh"""
    scale = 50 / np.max(magnitude) if np.max(magnitude) > 1e-6 else 100
    deformed = mesh_data['nodes'] + scale * displacement
    plt.triplot(*deformed.T, triangles=mesh_data['elements'],
               color='green', linewidth=0.8)
    plt.title(f"Deformed Configuration (Scale factor: {scale:.0f}x)")
    plt.axis('equal')

def validate_mesh(mesh_data):
    """execute mesh quality check"""
    # check element area
    assert np.all(mesh_data['element_area'] > 1e-10), "invalid element"
    
    # check unique nodes
    unique_nodes = np.unique(mesh_data['nodes'], axis=0)
    assert len(unique_nodes) == len(mesh_data['nodes']), "duplicate nodes"
    
    # check boundary pairing
    bord = mesh_data['boundary']
    assert len(bord['bottom']) == len(bord['top']), "bottom and top boundary nodes mismatch"
    assert len(bord['left']) == len(bord['right']), "left and right boundary nodes mismatch"
    
    print("mesh quality check passed")

if __name__ == "__main__":
    main()
