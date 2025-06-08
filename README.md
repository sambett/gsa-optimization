# Gravitational Search Algorithm (GSA) Implementation

This repository contains a Python implementation of the Gravitational Search Algorithm (GSA), a physics-inspired metaheuristic optimization algorithm. The implementation includes several benchmark functions and visualization tools for analyzing the algorithm's performance.

## Features

- Pure NumPy implementation of GSA
- Multiple benchmark functions (Sphere, Rastrigin, Rosenbrock, Ackley)
- Population size analysis tools
- Visualization of convergence behavior
- Comprehensive documentation and examples

## Installation

```bash
git clone https://github.com/yourusername/gsa-optimization.git
cd gsa-optimization
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from src.gsa import gravitational_search
import numpy as np

# Run GSA optimization
best_position, best_fitness = gravitational_search(
    n_particles=30,
    n_dimensions=30,
    fitness_function='sphere'
)
```

For more examples, check the `examples/` directory.
The examples show:

  - simple_example.py: Basic usage of GSA
  - compare_functions.py: How GSA performs on different optimization problems
  - population_study.py: How population size affects optimization performance

    ### Run basic example
``` bash
        python examples/simple_example.py

        # Compare different functions
        python examples/compare_functions.py

        # Study population sizes
        python examples/population_study.py
```
## Benchmark Functions

The implementation includes several standard benchmark functions:
- Sphere Function (unimodal)
- Rastrigin Function (multimodal)
- Rosenbrock Function (valley-shaped)
- Ackley Function (multimodal)

## Results

Example convergence analysis for different population sizes on the Rosenbrock function:
[![image](https://github.com/user-attachments/assets/c5aa44cd-a2ea-48d1-84c0-9bec74129abb)
]

## Requirements

- numpy
- matplotlib
- time

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
