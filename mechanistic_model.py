"""Functional response mechanistic model.
"""

import numpy as np
from scipy.linalg import expm
import pandas as pd


def a_parameter_creator(beta, x):
    r"""Compute parameter value from regression coefficients.

    This function assumes that the parameter value (a) is given by:

    .. math::
        a = \exp{\beta \cdot x}

    Parameters
    ----------
    beta : np.ndarray
        Regression coefficients
    x : np.ndarray
        Covariate values

    Returns
    -------
    float
        Parameter value
    """
    return np.exp(np.dot(beta, x))


def construct_matrix(n0, a, r):
    """
    Constructs a matrix based on the given parameters.

    Parameters:
        n0 (int): The dimension parameter.
        a (float): The scaling factor for `a_ns`.
        r (float): The constant value for `rs`.

    Returns:
        numpy.ndarray: The constructed matrix.
    """
    # Create vector a_ns
    a_ns = np.array([-(n0 - i) * a for i in range(n0 + 1)])

    # Create vector rs
    rs = np.full(n0 + 1, r)
    rs[0] = 0

    # new time varying method:
    # rs = [0, max r, ..., min r]

    # Create diagonal matrices
    top_left = np.diag(a_ns)
    top_right = np.diag(rs)
    bottom_right = np.diag(-rs)

    # Create bottom_left as a zero matrix
    bottom_left = np.zeros((n0 + 1, n0 + 1))

    # Populate bottom_left matrix
    k = 0
    for i in range(1, n0 + 1):
        for j in range(n0 + 1):
            if i - j == 1:
                bottom_left[i, j] = -a_ns[k]
                k += 1

    # Combine matrices to form A
    A_top = np.hstack((top_left, top_right))
    A_bottom = np.hstack((bottom_left, bottom_right))
    A = np.vstack((A_top, A_bottom))

    return A


def solve_system_p(t, n0, a, r, p_zero):
    """
    Solves the system and computes the resulting vector.

    Parameters:
        t (float): Time parameter.
        n0 (int): Dimension parameter.
        a (float): Scaling factor for the matrix.
        r (float): Constant value for the matrix.
        p_zero (numpy.ndarray): Initial state vector.

    Returns:
        numpy.ndarray: The resulting vector after solving the system.
    """
    # Dimension of the matrix
    dim_matrix = 2 * (n0 + 1)

    # Construct matrix A
    A = construct_matrix(n0, a, r)

    # Compute the matrix exponential of A * t
    exp_A = expm(A * t)

    # Compute the product exp_A * p_zero
    p_both_states = exp_A @ p_zero

    # Dimension of the output vector
    dim_p = n0 + 1

    # Extract and combine components from p_both_states
    p = np.zeros(dim_p)
    for i in range(dim_p):
        p[i] = p_both_states[i] + p_both_states[i + dim_p]  # Combine stacked system

    return p


def solve_system_simple_h_simplex(n_prey_initial, a, r):
    """
    Solve the system of equations for a given number of prey, attack rate, and 1/handling time.
    
    Args:
        n_prey_initial: Initial number of prey.
        a: Attack rate.
        r: 1/Handling time.
    
    Returns:
        A vector representing the solution probabilities.
    """
    # Initialize the initial probability vector
    p_zero = np.zeros(2 * (n_prey_initial + 1))
    p_zero[0] = 1  # Initial probability
    
    # Solve the system
    p = solve_system_p(1, n_prey_initial, a, r, p_zero)
    
    return p


def log_likelihood_single_fish(
    n_prey_eaten: int, n_prey_initial: np.ndarray, 
    beta: np.ndarray, gamma: np.ndarray, 
    x_a: np.ndarray, x_h: np.ndarray
) -> float:
    """
    Compute the log-likelihood for a single fish.
    
    Parameters:
        n_prey_eaten (int): Number of prey eaten.
        n_prey_initial (np.ndarray): Initial prey counts.
        beta (np.ndarray): Coefficients for 'a'.
        gamma (np.ndarray): Coefficients for 'h'.
        x_a (np.ndarray): Features for 'a'.
        x_h (np.ndarray): Features for 'h'.
    
    Returns:
        float: Log-likelihood for the single fish.
    """
    a = a_parameter_creator(beta, x_a)
    # should be a_parameter_creator here as the r parameter is bounded below at zero
    # note this is different to the other model
    r = a_parameter_creator(gamma, x_h)
    p_eaten = solve_system_simple_h_simplex(n_prey_initial, a, r)

    # Avoid obtaining negative values
    # TODO:  Check if this is needed. It seems to have a substantial effect on the loglikelihood values. Removed for now.
    # p_eaten[p_eaten<1e-10] = 1e-10

    return np.log(p_eaten[n_prey_eaten])  # Take the log for log-likelihood


def log_likelihood_all_fish(
    ns_prey_eaten: np.ndarray, ns_prey_initial: np.ndarray, 
    beta: np.ndarray, gamma: np.ndarray, 
    X_a: pd.DataFrame, X_h: pd.DataFrame
) -> float:
    """
    Compute the total log-likelihood for all fish.
    
    Parameters:
        ns_prey_eaten (np.ndarray): Array of prey eaten counts for all fish.
        ns_prey_initial (np.ndarray): Array of initial prey counts for all fish.
        beta (np.ndarray): Coefficients for 'a'.
        gamma (np.ndarray): Coefficients for 'h'.
        X_a (pd.DataFrame): Feature matrix for 'a'.
        X_h (pd.DataFrame): Feature matrix for 'h'.
    
    Returns:
        float: Total log-likelihood across all fish.
    """

    log_likelihood = 0.0
    fish_specific_log_likelihoods = []
    for n_prey_eaten, n_prey_initial, x_a, x_h in zip(
        ns_prey_eaten, ns_prey_initial, X_a.to_numpy(), X_h.to_numpy()
    ):
        ll = log_likelihood_single_fish(
            n_prey_eaten, n_prey_initial, beta, gamma, x_a, x_h
        )

        log_likelihood += ll
        
        # save fish-specific log likelihoods
        fish_specific_log_likelihoods.append(ll)

    return log_likelihood, fish_specific_log_likelihoods
