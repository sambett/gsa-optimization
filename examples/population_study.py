"""
Example studying the effect of population size on GSA performance
"""
from src.gsa import gravitational_search
import matplotlib.pyplot as plt
import numpy as np

def study_population_sizes():
    # Different population sizes to test
    pop_sizes = [10, 30, 50, 100]
    results = {}
    
    # Test each population size
    for pop in pop_sizes:
        print(f"\nTesting population size {pop}...")
        _, best_fitness, history = gravitational_search(
            n_particles=pop,
            n_dimensions=30,
            max_iter=1000,
            fitness_function='sphere'
        )
        results[pop] = history
        print(f"Best fitness: {best_fitness:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for pop, history in results.items():
        plt.plot(history, label=f'Population = {pop}')
    
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Effect of Population Size on GSA Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    study_population_sizes()
