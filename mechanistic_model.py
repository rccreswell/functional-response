"""Slower, mechanistic model.
"""

import numpy as np
from scipy.linalg import expm
import pandas as pd

import likelihood


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
    a = likelihood.a_parameter_creator(beta, x_a)
    # should be a_parameter_creator here as the r parameter is bounded below at zero
    # note this is different to the above model
    r = likelihood.a_parameter_creator(gamma, x_h)
    p_eaten = solve_system_simple_h_simplex(n_prey_initial, a, r)

    # print(a, r)

    ### Obtaining negative values
    p_eaten[p_eaten<1e-10] = 1e-10

    return np.log(p_eaten[n_prey_eaten])  # Take the log for log-likelihood


def log_likelihood_single_fish_ar(
    n_prey_eaten: int, n_prey_initial: np.ndarray, 
    a, r,
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
    p_eaten = solve_system_simple_h_simplex(n_prey_initial, a, r)

    # print(a, r)

    ### Obtaining negative values
    p_eaten[p_eaten<1e-10] = 1e-10

    return np.log(p_eaten[n_prey_eaten])  # Take the log for log-likelihood


def log_likelihood_single_fish_ar_fast(
    n_prey_eaten: int, n_prey_initial: np.ndarray, 
    a, r, fast_model       
):
    p_eaten = fast_model[n_prey_initial].predict([[a, r]])[0]

    p_eaten[p_eaten<1e-10] = 1e-10

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
    for n_prey_eaten, n_prey_initial, x_a, x_h in zip(
        ns_prey_eaten, ns_prey_initial, X_a.to_numpy(), X_h.to_numpy()
    ):
        log_likelihood += log_likelihood_single_fish(
            n_prey_eaten, n_prey_initial, beta, gamma, x_a, x_h
        )
        
        # save fish-specific log likelihoods
        

    return log_likelihood


def log_likelihood_all_fish_ar(
    ns_prey_eaten: np.ndarray, ns_prey_initial: np.ndarray, 
    a, r,
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
    for n_prey_eaten, n_prey_initial in zip(
        ns_prey_eaten, ns_prey_initial
    ):
        log_likelihood += log_likelihood_single_fish_ar(
            n_prey_eaten, n_prey_initial, a, r
        )
    return log_likelihood


def log_likelihood_all_fish_ar_weights(
    ns_prey_eaten: np.ndarray, ns_prey_initial: np.ndarray, 
    a, r_constant, r_trend, weights, fast_model=None
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
    for n_prey_eaten, n_prey_initial, we in zip(
        ns_prey_eaten, ns_prey_initial, weights
    ):
        if fast_model is None:
            log_likelihood += log_likelihood_single_fish_ar(
            n_prey_eaten, n_prey_initial, a, np.exp(r_constant + r_trend * we)
         )
        else:
            log_likelihood += log_likelihood_single_fish_ar_fast(
                n_prey_eaten, n_prey_initial, a, np.exp(r_constant + r_trend * we), fast_model
            )
    return log_likelihood



def log_likelihood_all_fish_ar_weights_exp(
    ns_prey_eaten: np.ndarray, ns_prey_initial: np.ndarray, 
    a, r_constant, r_trend, condH, condH6, weights, acc_condH, acc_condH_6,
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
    for n_prey_eaten, n_prey_initial, we, acc_condH_i, acc_condH_6_i in zip(
        ns_prey_eaten, ns_prey_initial, weights, acc_condH, acc_condH_6
    ):
        log_likelihood += log_likelihood_single_fish_ar(
            n_prey_eaten, n_prey_initial, a, np.exp(r_constant + r_trend * we + condH * acc_condH_i + condH6 * acc_condH_6_i)
        )
    return log_likelihood


def create_parameter_values(as_values=None, rs_values=None):

    if as_values is None:
        # User-defined parameters
        sigma_a = 0.1  # Standard deviation for noise added to as_values
        as_values= np.exp(np.linspace(np.log(0.05), np.log(600), 200))  # Exponential sequence
        as_values = as_values + np.random.normal(0, sigma_a, len(as_values))
        as_values = np.maximum(as_values, 1e-8)  # Ensure all values > 0

    if rs_values is None:
        sigma_h = 0.02  # Standard deviation for noise added to hs_values
        hs_values = np.exp(np.linspace(np.log(0.01), np.log(5), 200))  # Exponential sequence
        hs_values = hs_values + np.random.normal(0, sigma_h, len(hs_values))
        rs_values = np.clip(hs_values, 0, 1)  # Ensure values between 0 and 1

    # Create a grid of all combinations
    parameters = pd.DataFrame(
        [(a, h,) for a in as_values for h in rs_values],
        columns=['a', 'r',]
    )

    return parameters


def create_training_data(ar_parameters, n):
    
    m_simplices = np.zeros((len(ar_parameters), n + 1))

    # Loop through each row in parameters
    for i, row in ar_parameters.iterrows():
        # Solve the system
        ps = solve_system_simple_h_simplex(
            n, 
            row['a'], 
            row['r']
        )
        
        # Convert to numpy array and handle numerical underflow
        ps = np.array(ps)
        ps[ps < 0] = abs(ps[ps < 0])  # Replace negative values with their absolute
        
        # Pad with zeros if needed
        if len(ps) < n + 1:
            ps = np.pad(ps, (0, n + 1 - len(ps)), constant_values=0)
        
        # Assign to matrix
        m_simplices[i, :] = ps

    # Check row sums
    rsums = m_simplices.sum(axis=1)
    assert np.isclose(rsums.sum(), len(m_simplices)), "Sum of row sums does not match number of rows"

    return m_simplices




def fast_log_likelihood_all_fish(ns_prey_eaten, ns_prey_initial, beta, gamma, X_a, X_h, models):
    log_likelihood = 0.0

    all_as = []
    all_rs = []
    all_ns = []
    for n_prey_eaten, n_prey_initial, x_a, x_h in zip(
        ns_prey_eaten, ns_prey_initial, X_a.to_numpy(), X_h.to_numpy()
    ):
        a = fastprey.a_parameter_creator(beta, x_a)
        r = fastprey.a_parameter_creator(gamma, x_h)

        # print(a, r)

        all_as.append(a)
        all_rs.append(r)
        all_ns.append(n_prey_initial)

    all_as = np.asarray(all_as)[:, np.newaxis]
    all_rs = np.asarray(all_rs)[:, np.newaxis]
    all_ns = np.asarray(all_ns)

    i = 0

    for n in sorted(list(set(all_ns))):

        a = all_as[np.where(all_ns == n)]
        r = all_rs[np.where(all_ns == n)]

        


        simplices = models[n].predict(np.concatenate((a, r), axis=1))

        for simplex in simplices:
            n_prey_eaten = ns_prey_eaten[i]
            i += 1
            
            log_likelihood += fast_log_likelihood_single_fish(
                    n_prey_eaten, simplex
            )

    return log_likelihood


def fast_log_likelihood_single_fish(n_prey_eaten, simplex):

    ### Obtaining negative values
    simplex[simplex<1e-10] = 1e-10

    return np.log(simplex[n_prey_eaten])