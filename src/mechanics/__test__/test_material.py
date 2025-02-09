import numpy as np
from src.mechanics.material import Material

def test_material_initialization():
    """Test material properties initialization with plane strain assumption"""
    E = [1.0, 10.0]
    nu = [0.3, 0.3]
    material = Material(E, nu, plane_stress=False)
    
    # Check basic properties
    assert np.allclose(material.youngs_modulus, np.array(E))
    assert np.allclose(material.poissons_ratio, np.array(nu))
    
    # Verify Lame parameters calculation under plane strain
    expected_lambd = (np.array(E) * np.array(nu)) / ((1 + np.array(nu)) * (1 - 2 * np.array(nu)))
    expected_mu = np.array(E) / (2 * (1 + np.array(nu)))
    
    assert np.allclose(material.lambd, expected_lambd)
    assert np.allclose(material.mu, expected_mu)

def test_constitutive_matrix():
    """Test constitutive matrix generation for plane strain"""
    E = [1.0, 10.0]
    nu = [0.3, 0.3]
    material = Material(E, nu, plane_stress=False)
    
    C = material.get_constitutive_matrix()
    
    # Check matrix dimensions
    assert C.shape == (2, 3, 3)
    
    # Verify symmetry
    for i in range(2):
        assert np.allclose(C[i], C[i].T)
    
    # Validate plane strain constitutive matrix values
    for i in range(2):
        expected_C = np.array([
            [material.lambd[i] + 2*material.mu[i], material.lambd[i], 0],
            [material.lambd[i], material.lambd[i] + 2*material.mu[i], 0],
            [0, 0, material.mu[i]]
        ])
        assert np.allclose(C[i], expected_C)

def test_plane_stress_case():
    """Test material properties under plane stress condition"""
    E = [1.0, 10.0]
    nu = [0.3, 0.3]
    material = Material(E, nu, plane_stress=True)
    
    # Verify Lame parameters in plane stress
    expected_lambd = (np.array(E) * np.array(nu)) / (1 - np.array(nu)**2)
    expected_mu = np.array(E) / (2 * (1 + np.array(nu)))
    
    assert np.allclose(material.lambd, expected_lambd)
    assert np.allclose(material.mu, expected_mu)
    
    # Validate plane stress constitutive matrix
    C = material.get_constitutive_matrix()
    for i in range(2):
        factor = E[i] / (1 - nu[i]**2)
        expected_C = factor * np.array([
            [1, nu[i], 0],
            [nu[i], 1, 0],
            [0, 0, (1 - nu[i])/2]
        ])
        assert np.allclose(C[i], expected_C)
