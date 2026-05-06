"""The true model.
"""

import numpy as np
from scipy.linalg import expm


def rogersII(n_prey_remaining, a, h):
    """Rogers type II functional response."""
    return a * n_prey_remaining / (1 + a * h * n_prey_remaining)


def construct_matrix(lambdas, dim_matrix):
    """
    Construct a matrix for the system of equations.
    
    Args:
        lambdas: A list of lambda values.
        dim_matrix: The dimension of the matrix.
    
    Returns:
        A square matrix of size dim_matrix x dim_matrix.
    """
    # Initialize the matrix with zeros
    A = np.zeros((dim_matrix, dim_matrix))
    
    # Set the first element
    A[0, 0] = -lambdas[0]
    
    # Loop to fill in the middle part of the matrix
    for i in range(1, dim_matrix - 1):
        A[i, i - 1] = lambdas[i - 1]
        A[i, i] = -lambdas[i]
    
    # Set the last element
    A[dim_matrix - 1, dim_matrix - 2] = lambdas[dim_matrix - 2]
    
    return A


def solve_system_p(t, lambdas, p_zero):
    """
    Solve the system of equations using matrix exponentials.
    
    Args:
        t: Time step for the matrix exponential.
        lambdas: A list of lambda values.
        p_zero: Initial probability distribution vector.
    
    Returns:
        A vector representing the solution of the system.
    """
    dim_matrix = len(lambdas) + 1
    A = construct_matrix(lambdas, dim_matrix)
    
    # Calculate matrix exponential
    exp_A = expm(A * t)
    
    # Calculate p
    p = np.dot(exp_A, p_zero)
    
    return p


def solve_system_simple_h_simplex(n_prey_initial, a, h):
    """
    Solve the system of equations for a given number of prey, attack rate, and handling time.
    
    Args:
        n_prey_initial: Initial number of prey.
        a: Attack rate.
        h: Handling time.
    
    Returns:
        A vector representing the solution probabilities.
    """
    # Initialize the list of possible prey remaining
    n_possible_prey_remaining = [n_prey_initial - i + 1 for i in range(1, n_prey_initial + 1)]
    
    # Initialize the initial probability vector
    p_zero = np.zeros(n_prey_initial + 1)
    p_zero[0] = 1  # Initial probability
    
    # Calculate lambda values
    lambdas = [rogersII(n, a, h) for n in n_possible_prey_remaining]
    
    # Solve the system
    p = solve_system_p(1, lambdas, p_zero)
    
    return p