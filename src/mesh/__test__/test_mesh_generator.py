import numpy as np
import pytest
from ..mesh_generator import MeshGenerator
import unittest

class TestMeshGenerator(unittest.TestCase):
    # Before: 测试网格生成基础功能
    # After: Test basic mesh generation functionality
    def test_mesh_generation(self):
        # Before: 验证节点和单元数量
        # After: Verify node and element counts
        length = 1.0
        nx = 4
        mesh_gen = MeshGenerator(length, nx)
        mesh = mesh_gen.generate_mesh()
        
        # Test if all required keys are present
        required_keys = ['nodes', 'elements', 'dof_map', 'element_centers', 'boundary']
        for key in required_keys:
            assert key in mesh
        
        # Test mesh dimensions
        n_nodes = nx * nx
        assert mesh['nodes'].shape == (n_nodes, 2)
        
        # Test element connectivity
        assert mesh['elements'].shape[1] == 3  # triangular elements
        
        # Test boundary nodes
        boundary = mesh['boundary']
        assert all(key in boundary for key in ['bottom', 'right', 'top', 'left'])
        
        # Test boundary node counts
        n = nx
        assert len(boundary['bottom']) == n
        assert len(boundary['right']) == n
        assert len(boundary['top']) == n
        assert len(boundary['left']) == n

    def test_mesh_coordinates():
        """Test if mesh coordinates are correctly centered and scaled"""
        length = 2.0
        nx = 4
        mesh_gen = MeshGenerator(length, nx)
        mesh = mesh_gen.generate_mesh()
        
        nodes = mesh['nodes']
        
        # Test if mesh is centered at origin
        assert np.allclose(np.mean(nodes, axis=0), [0, 0])
        
        # Test if mesh has correct dimensions
        assert np.allclose(np.max(nodes[:, 0]) - np.min(nodes[:, 0]), length)
        assert np.allclose(np.max(nodes[:, 1]) - np.min(nodes[:, 1]), length)
