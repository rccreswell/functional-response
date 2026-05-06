"""Computing the likelihood.
"""

import numpy as np
import matplotlib.pyplot as plt

import model

def a_parameter_creator(beta, x):
    return np.exp(np.dot(beta, x))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def h_parameter_creator(gamma, x):
    return logistic(np.dot(gamma, x))


def log_likelihood_single_fish(n_prey_eaten, n_prey_initial, beta, gamma, x_a, x_h):
    a = a_parameter_creator(beta, x_a)
    h = h_parameter_creator(gamma, x_h)
    p_eaten = model.solve_system_simple_h_simplex(n_prey_initial, a, h)
    return np.log(p_eaten[n_prey_eaten])
    

def log_likelihood_all_fish(ns_prey_eaten, ns_prey_initial, beta, gamma, X_a, X_h):
    log_likelihood = 0.0
    for n_prey_eaten, n_prey_initial, x_a, x_h in zip(
        ns_prey_eaten, ns_prey_initial, X_a.to_numpy(), X_h.to_numpy()
    ):
        log_likelihood += log_likelihood_single_fish(
                n_prey_eaten, n_prey_initial, beta, gamma, x_a, x_h
        )

    return log_likelihood


def log_likelihood_single_fish_ah(n_prey_eaten, n_prey_initial, a, h):
    p_eaten = model.solve_system_simple_h_simplex(n_prey_initial, a, h)

 ### Obtaining negative values
    p_eaten[p_eaten<1e-10] = 1e-10

    return np.log(p_eaten[n_prey_eaten])
    

def log_likelihood_all_fish_ah(ns_prey_eaten, ns_prey_initial, a, h):
    log_likelihood = 0.0
    for n_prey_eaten, n_prey_initial in zip(
        ns_prey_eaten, ns_prey_initial
    ):
        log_likelihood += log_likelihood_single_fish_ah(
                n_prey_eaten, n_prey_initial, a, h
        )

    return log_likelihood



def log_likelihood_all_fish_ah_weights(ns_prey_eaten, ns_prey_initial, a, h_constant, h_trend, weights):
    log_likelihood = 0.0
    for n_prey_eaten, n_prey_initial, we in zip(
        ns_prey_eaten, ns_prey_initial, weights
    ):
        log_likelihood += log_likelihood_single_fish_ah(
                n_prey_eaten, n_prey_initial, a, h_constant + h_trend * we
        )

    return log_likelihood



def fast_log_likelihood_all_fish(ns_prey_eaten, ns_prey_initial, beta, gamma, X_a, X_h, models):
    log_likelihood = 0.0

    all_as = []
    all_hs = []
    all_ns = []
    for n_prey_eaten, n_prey_initial, x_a, x_h in zip(
        ns_prey_eaten, ns_prey_initial, X_a.to_numpy(), X_h.to_numpy()
    ):
        a = a_parameter_creator(beta, x_a)
        h = h_parameter_creator(gamma, x_h)

        all_as.append(a)
        all_hs.append(h)
        all_ns.append(n_prey_initial)

    all_as = np.asarray(all_as)[:, np.newaxis]
    all_hs = np.asarray(all_hs)[:, np.newaxis]
    all_ns = np.asarray(all_ns)

    i = 0

    for n in sorted(list(set(all_ns))):

        a = all_as[np.where(all_ns == n)]
        h = all_hs[np.where(all_ns == n)]

        simplices = models[n].predict(np.concatenate((a, h), axis=1))

        for simplex in simplices:
            n_prey_eaten = ns_prey_eaten[i]
            i += 1
            
            log_likelihood += fast_log_likelihood_single_fish(
                    n_prey_eaten, simplex
            )

    return log_likelihood


def fast_log_likelihood_single_fish(n_prey_eaten, simplex):
    return np.log(simplex[n_prey_eaten])