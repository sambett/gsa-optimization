import numpy as np
from typing import Tuple, List, Optional
from .benchmark_functions import calculate_fitness

class GSA:
    """Gravitational Search Algorithm implementation"""
    
    def __init__(
        self,
        n_particles: int = 30,
        n_dimensions: int = 30,
        max_iter: int = 1000,
        G0: float = 100,
        alpha: float = 20,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        fitness_function: str = 'sphere'
    ):
        """Initialize GSA optimizer
        
        Args:
            n_particles: Number of agents
            n_dimensions: Problem dimensions
            max_iter: Maximum iterations
            G0: Initial gravitational constant
            alpha: Decay rate for gravitational constant
            bounds: Search space bounds (min, max)
            fitness_function: Objective function to optimize
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iter = max_iter
        self.G0 = G0
        self.alpha = alpha
        self.bounds = bounds
        self.fitness_function = fitness_function
        
        # Initialize arrays
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_fitness = np.inf
        self.fitness_history = []
        
    def _calculate_masses(self, fitness: np.ndarray) -> np.ndarray:
        """Calculate masses based on fitness values"""
        worst = np.max(fitness)
        best = np.min(fitness)
        masses = (worst - fitness) / (worst - best + 1e-10)
        return masses / (np.sum(masses) + 1e-10)
    
    def _calculate_acceleration(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        G: float,
        k_best: int
    ) -> np.ndarray:
        """Calculate acceleration for each agent based on gravitational forces"""
        best_idx = np.argsort(masses)[-k_best:]
        
        diff_matrix = positions[best_idx][:, np.newaxis] - positions
        distances = np.sqrt(np.sum(diff_matrix ** 2, axis=2) + 1e-10)
        
        masses_product = masses[best_idx][:, np.newaxis] * masses[np.newaxis, :]
        force_terms = G * masses_product[:, :, np.newaxis] * diff_matrix
        force_terms = force_terms / distances[:, :, np.newaxis]
        
        acceleration = np.sum(force_terms, axis=0)
        return acceleration / (masses[:, np.newaxis] + 1e-10)
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run the GSA optimization process
        
        Returns:
            Tuple containing:
            - Best position found
            - Best fitness value
            - History of best fitness values
        """
        # Initialize positions and velocities
        self.positions = np.random.uniform(
            self.bounds[0],
            self.bounds[1],
            (self.n_particles, self.n_dimensions)
        )
        self.velocities = np.zeros((self.n_particles, self.n_dimensions))
        
        # Pre-calculate gravitational constant decay
        G_values = self.G0 * np.exp(-self.alpha * np.arange(self.max_iter) / self.max_iter)
        
        for t in range(self.max_iter):
            # Calculate fitness and update best solution
            fitness = calculate_fitness(self.positions, self.fitness_function)
            min_idx = np.argmin(fitness)
            
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_position = self.positions[min_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Update positions and velocities
            masses = self._calculate_masses(fitness)
            k_best = max(1, int(self.n_particles * (1 - t / self.max_iter)))
            acceleration = self._calculate_acceleration(
                self.positions, masses, G_values[t], k_best
            )
            
            self.velocities = (np.random.random((self.n_particles, self.n_dimensions)) * 
                             self.velocities + acceleration)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
        
        return self.best_position, self.best_fitness, self.fitness_history

def gravitational_search(*args, **kwargs):
    """Convenience function to run GSA optimization"""
    optimizer = GSA(*args, **kwargs)
    return optimizer.optimize()
