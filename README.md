# Finite Element Analysis for Composite Materials

A Python-based finite element analysis tool for simulating mechanical behavior of composite materials with periodic boundary conditions.

## Features

- 2D linear elastic analysis
- Periodic boundary conditions
- Multi-phase material support
- Adaptive mesh generation
- Numerical stability handling
- Visualization of results

## Environment Setup

### Prerequisites
- Python 3.8+
- [Rye](https://rye-up.com/) package manager

### Installation

Clone repository
git clone https://github.com/yourusername/finite-element-composite.git
cd finite-element-composite
Install dependencies
rye sync
Structure

## Project Structure

```
├── src/
│   ├── mechanics/          # Material properties and constitutive models
│   │   └── material.py
│   ├── mesh/               # Mesh generation and processing
│   │   └── mesh_generator.py
│   └── solver/             # Finite element solver implementation
│       └── fem_solver.py
├── main.py                 # Main simulation workflow
├── CONFIG                  # Simulation parameters
└── requirements.txt        # Dependency list
```

## Configuration
Edit `main.py` to modify simulation parameters:
```python
CONFIG = {
    "length": 1.0,         # Domain size (meters)
    "nx": 60,              # Mesh density (elements per side)
    "inclusions": 2,       # Number of inclusions
    "radius": 0.2,         # Inclusion radius (meters)
    "material_props": {
        "E": [1e9, 50e9],  # Young's modulus values (Pa)
        "nu": [0.25, 0.3]  # Poisson's ratios
    },
    "applied_force": 5e4,  # Applied tensile force (N)
    "plane_stress": True   # Analysis type
}
```

## Running the Simulation
```bash
# Run main simulation
python main.py

# Expected output:
# - Material phase visualization
# - Displacement contour plot
# - Deformed mesh visualization
# - Console output with numerical diagnostics
```

## Key Components

### Material Model (`src/mechanics/material.py`)
- Implements plane stress/strain constitutive laws
- Validates material properties
- Calculates Lame parameters
- Generates constitutive matrices

### Mesh Generator (`src/mesh/mesh_generator.py`)
- Generates structured quadrilateral meshes
- Calculates element areas and centroids
- Identifies boundary nodes
- Creates degree-of-freedom mapping

### FEM Solver (`src/solver/fem_solver.py`)
- Assembles global stiffness matrix
- Implements periodic boundary conditions
- Solves linear systems with regularization
- Handles numerical stability issues

## Visualization
The simulation generates three plots:
1. Material phase distribution
2. Displacement magnitude contour
3. Deformed mesh configuration

![Sample Output](docs/sample_output.png)

## Troubleshooting
Common issues and solutions:

**Q: Getting NaN values in displacement results**
- Verify material properties are physically valid
- Check boundary condition constraints
- Reduce applied load magnitude

**Q: Mesh quality warnings**
- Increase mesh density (`nx` in CONFIG)
- Check inclusion radius vs domain size

**Q: Singular matrix errors**
- Confirm proper boundary constraints
- Enable plane stress condition
- Increase regularization in solver

## License
MIT License - See [LICENSE](LICENSE) for details.
