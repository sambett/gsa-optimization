import numpy as np
import pytest
from src.gsa import GSA
from src.benchmark_functions import calculate_fitness

def test_gsa_initialization():
    optimizer = GSA()
    assert optimizer.n_particles == 30
    assert optimizer.n_dimensions == 30
    assert optimizer.best_fitness == np.inf

def test_sphere_function_optimization():
    optimizer = GSA(
        n_particles=20,
        n_dimensions=10,
        max_iter=100,
        fitness_function='sphere'
    )
    best_position, best_fitness, _ = optimizer.optimize()
    
    assert best_fitness < 1.0  # Should easily find good solution for sphere
    assert best_position.shape == (10,)

def test_invalid_fitness_function():
    with pytest.raises(ValueError):
        GSA(fitness_function='invalid')
