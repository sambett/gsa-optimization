import numpy as np

def sphere_function(positions):
    """Sphere function - simple unimodal function
    
    Args:
        positions (np.ndarray): Position vectors of shape (n_particles, n_dimensions)
    
    Returns:
        np.ndarray: Fitness values for each position
        
    Global minimum: f(0,...,0) = 0
    """
    return np.sum(positions ** 2, axis=1)

def rastrigin_function(positions):
    """Rastrigin function - highly multimodal function
    
    Args:
        positions (np.ndarray): Position vectors of shape (n_particles, n_dimensions)
    
    Returns:
        np.ndarray: Fitness values for each position
        
    Global minimum: f(0,...,0) = 0
    """
    n = positions.shape[1]
    return 10 * n + np.sum(positions ** 2 - 10 * np.cos(2 * np.pi * positions), axis=1)

def rosenbrock_function(positions):
    """Rosenbrock function (banana function) - hard to optimize
    
    Args:
        positions (np.ndarray): Position vectors of shape (n_particles, n_dimensions)
    
    Returns:
        np.ndarray: Fitness values for each position
        
    Global minimum: f(1,...,1) = 0
    """
    return np.sum(100 * (positions[:, 1:] - positions[:, :-1]**2) ** 2 + 
                 (1 - positions[:, :-1]) ** 2, axis=1)

def ackley_function(positions):
    """Ackley function - multimodal function with many local minima
    
    Args:
        positions (np.ndarray): Position vectors of shape (n_particles, n_dimensions)
    
    Returns:
        np.ndarray: Fitness values for each position
        
    Global minimum: f(0,...,0) = 0
    """
    n = positions.shape[1]
    sum1 = np.sum(positions ** 2, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * positions), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def calculate_fitness(positions, function='sphere'):
    """Calculate fitness using the selected function
    
    Args:
        positions (np.ndarray): Position vectors
        function (str): Name of the fitness function to use
        
    Returns:
        np.ndarray: Fitness values for each position
    
    Raises:
        ValueError: If unknown fitness function is specified
    """
    function_map = {
        'sphere': sphere_function,
        'rastrigin': rastrigin_function,
        'rosenbrock': rosenbrock_function,
        'ackley': ackley_function
    }
    
    if function not in function_map:
        raise ValueError(f"Unknown fitness function: {function}")
    
    return function_map[function](positions)
