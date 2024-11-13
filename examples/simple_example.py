"""
Simple example showing how to use the GSA implementation
"""
import numpy as np
from src.gsa import gravitational_search
import matplotlib.pyplot as plt

def run_basic_example():
    # Run GSA optimization on sphere function
    best_position, best_fitness, history = gravitational_search(
        n_particles=30,  # number of agents
        n_dimensions=30, # problem dimensions
        max_iter=1000,   # maximum iterations
        fitness_function='sphere' # which function to optimize
    )

    # Print results
    print(f"Best fitness found: {best_fitness:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('GSA Convergence')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_basic_example()
