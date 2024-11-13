import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from .gsa import gravitational_search

def run_population_analysis(
    population_sizes: List[int] = [10, 30, 40, 80, 100],
    n_trials: int = 5,
    fitness_function: str = 'rosenbrock'
) -> Dict:
    """Run analysis with different population sizes and visualize results
    
    Args:
        population_sizes: List of population sizes to test
        n_trials: Number of trials for each population size
        fitness_function: Benchmark function to optimize
        
    Returns:
        Dictionary containing results for each population size
    """
    results = {pop_size: [] for pop_size in population_sizes}
    global_best_fitness = np.inf
    
    for pop_size in population_sizes:
        print(f"\nTesting population size: {pop_size}")
        for trial in range(n_trials):
            start_time = time.time()
            
            _, best_fitness, fitness_history = gravitational_search(
                n_particles=pop_size,
                fitness_function=fitness_function
            )
            
            results[pop_size].append(fitness_history)
            
            if best_fitness < global_best_fitness:
                global_best_fitness = best_fitness
            
            print(f"Trial {trial + 1}: Time={time.time() - start_time:.3f}s, "
                  f"Best fitness={best_fitness:.5f}")
    
    return results

def plot_convergence(results: Dict, title: str = "GSA Convergence Analysis"):
    """Create convergence plot from analysis results
    
    Args:
        results: Dictionary containing results for each population size
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (pop_size, trials), color in zip(results.items(), colors):
        trials = np.array(trials)
        mean = np.mean(trials, axis=0)
        std = np.std(trials, axis=0)
        
        plt.plot(mean, label=f'Pop={pop_size}', color=color, linewidth=2)
        plt.fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            color=color,
            alpha=0.2
        )
    
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness (log scale)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()
