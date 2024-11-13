"""
Example comparing GSA performance on different benchmark functions
"""
import numpy as np
from src.gsa import gravitational_search
import matplotlib.pyplot as plt

def compare_benchmark_functions():
    # List of functions to test
    functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley']
    results = {}
    
    # Test each function
    for func in functions:
        print(f"\nTesting {func} function...")
        best_position, best_fitness, history = gravitational_search(
            n_particles=30,
            n_dimensions=30,
            max_iter=1000,
            fitness_function=func
        )
        results[func] = history
        print(f"Best fitness: {best_fitness:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for func, history in results.items():
        plt.plot(history, label=func)
    
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('GSA Performance on Different Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_benchmark_functions()
