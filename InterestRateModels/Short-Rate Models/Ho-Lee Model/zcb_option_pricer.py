import numpy as np
from scipy.stats import norm

from helpers import matrix_L_ho_lee, ho_lee_lattice


#====================================================
# ZCB Option Valuation: Ho-Lee Closed Form
#====================================================
def zcb_option_ho_lee_closed_form(sigma, t, T_1, T_2, K, P0, option_type="call", N=1.0):
    if not t < T_1 < T_2:
        raise ValueError("Require t < T_1 < T_2.")

    P_t_T1 = P0(T_1) / P0(t)
    P_t_T2 = P0(T_2) / P0(t)

    Sigma = sigma * (T_2 - T_1) * np.sqrt(T_1 - t)

    if Sigma <= 0.0:
        if option_type == "call":
            price = N * max(P_t_T2 - K * P_t_T1, 0.0)
        elif option_type == "put":
            price = N * max(K * P_t_T1 - P_t_T2, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        return price

    d1 = (np.log(P_t_T2 / (K * P_t_T1)) + 0.5 * Sigma**2) / Sigma
    d2 = d1 - Sigma

    if option_type == "call":
        price = N * (P_t_T2 * norm.cdf(d1) - K * P_t_T1 * norm.cdf(d2))
    elif option_type == "put":
        price = N * (K * P_t_T1 * norm.cdf(-d2) - P_t_T2 * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return price


#====================================================
# ZCB Option Valuation: Ho-Lee Monte Carlo
#====================================================
def zcb_option_ho_lee_mc(r_paths, sigma, t, T_1, T_2, K, P0, f0_T1, option_type="call", N=1.0):
    if not t < T_1 < T_2:
        raise ValueError("Require t < T_1 < T_2.")

    n_time = r_paths.shape[1] - 1
    dt = (T_1 - t) / n_time

    integral_r = np.trapezoid(r_paths, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)

    r_T1 = r_paths[:, -1]
    B = T_2 - T_1

    P0_T1 = P0(T_1)
    P0_T2 = P0(T_2)

    sigma_term = 0.5 * sigma**2 * T_1 * B**2
    log_A = np.log(P0_T2 / P0_T1) + B * f0_T1 - sigma_term

    P_T1_T2 = np.exp(log_A - B * r_T1)

    if option_type == "call":
        payoff = N * np.maximum(P_T1_T2 - K, 0.0)
    elif option_type == "put":
        payoff = N * np.maximum(K - P_T1_T2, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    price = np.mean(discount_factors * payoff)
    return price


#====================================================
# ZCB Option Valuation: Ho-Lee Implicit FDM
#====================================================
def zcb_option_ho_lee_fdm(r0, theta, sigma, T_1, T_2, K, P0, f0_T1, option_type="call", Nr=301, Nt=500, r_min=-0.10, r_max=0.20, N=1.0):
    if not 0.0 < T_1 < T_2:
        raise ValueError("Require 0 < T_1 < T_2.")

    r_grid = np.linspace(r_min, r_max, Nr)
    dt = T_1 / Nt
    I = np.eye(Nr)

    B = T_2 - T_1
    P0_T1 = P0(T_1)
    P0_T2 = P0(T_2)

    sigma_term = 0.5 * sigma**2 * T_1 * B**2
    log_A = np.log(P0_T2 / P0_T1) + B * f0_T1 - sigma_term
    P_T1_T2 = np.exp(log_A - B * r_grid)

    if option_type == "call":
        V = N * np.maximum(P_T1_T2 - K, 0.0)
    elif option_type == "put":
        V = N * np.maximum(K - P_T1_T2, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    for n in range(Nt - 1, -1, -1):
        time = n * dt
        theta_t = theta(time) if callable(theta) else theta

        L = matrix_L_ho_lee(Nr, r_grid, theta_t, sigma)
        A = I - dt * L

        tau = T_1 - time

        if option_type == "call":
            V_lower = N * max(np.exp(log_A - B * r_grid[0]) - K, 0.0) * np.exp(-r_grid[0] * tau)
            V_upper = 0.0
        else:
            V_lower = 0.0
            V_upper = N * K * np.exp(-r_grid[-1] * tau)

        V[0] = V_lower
        V[-1] = V_upper

        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0

        V = np.linalg.solve(A, V)

    price = np.interp(r0, r_grid, V)
    return price


#====================================================
# ZCB Option Valuation: Ho-Lee Lattice
#====================================================
def zcb_option_ho_lee_lattice(sigma, T_1, T_2, K, P0, option_type="call", n_steps=500, N=1.0):
    if not 0.0 < T_1 < T_2:
        raise ValueError("Require 0 < T_1 < T_2.")

    dt = T_2 / n_steps
    n_expiry = int(round(T_1 / dt))

    if not np.isclose(n_expiry * dt, T_1):
        raise ValueError("T_1 must lie exactly on the lattice time grid.")

    r_tree, _, _ = ho_lee_lattice(sigma=sigma, T=T_2, n_steps=n_steps, P0=P0)

    # ------------------------------------------------
    # Stage 1: Price underlying ZCB from T_2 to T_1
    # ------------------------------------------------
    V_bond = np.ones(n_steps + 1)

    for n in range(n_steps - 1, n_expiry - 1, -1):
        V_new = np.zeros(n + 1)

        for j in range(n + 1):
            continuation = 0.5 * V_bond[j] + 0.5 * V_bond[j + 1]
            V_new[j] = np.exp(-r_tree[n, j] * dt) * continuation

        V_bond = V_new

    # ------------------------------------------------
    # Option payoff at T_1
    # ------------------------------------------------
    if option_type == "call":
        V = N * np.maximum(V_bond - K, 0.0)
    elif option_type == "put":
        V = N * np.maximum(K - V_bond, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    # ------------------------------------------------
    # Stage 2: Price option from T_1 to time 0
    # ------------------------------------------------
    for n in range(n_expiry - 1, -1, -1):
        V_new = np.zeros(n + 1)

        for j in range(n + 1):
            continuation = 0.5 * V[j] + 0.5 * V[j + 1]
            V_new[j] = np.exp(-r_tree[n, j] * dt) * continuation

        V = V_new

    price = V[0]
    return price