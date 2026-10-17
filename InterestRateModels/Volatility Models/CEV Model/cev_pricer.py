import numpy as np
from scipy.stats import norm, ncx2
from scipy.linalg import solve
from cev_helpers import *


#================ European Option on Forward Valuation: CEV Model =================

#=====================================================
# CEV European Option Valuation: Analytical Method
#=====================================================
def cev_option_price_analytical(
    F0, X, r, sigma, beta, T,
    option_type="call"
):
    if np.isclose(beta, 2.0):
        raise ValueError("beta cannot be equal to 2")
    if option_type.lower() not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    m = 2 - beta
    k = 2 / (sigma**2 * m**2 * T)
    x = k * F0**m
    y = k * X**m
    if beta < 2:
        call = np.exp(-r * T) * (
            F0 * ncx2.sf(2 * y, 2 + 2 / m, 2 * x)
            - X * ncx2.cdf(2 * x, 2 / m, 2 * y)
        )
    else:
        call = np.exp(-r * T) * (
            F0 * ncx2.sf(2 * x, 2 / (-m), 2 * y)
            - X * ncx2.cdf(2 * y, 2 + 2 / (-m), 2 * x)
        )
    if option_type.lower() == "call":
        return float(call)
    put = call - np.exp(-r * T) * (F0 - X)
    return float(put)


#========================================================
# CEV European Option Valuation: Monte Carlo Simulation
#========================================================
def cev_option_price_mc(
    F0, X, r, sigma, beta, T,
    n_time, n_sim,
    option_type="call", seed=None
):
    F_paths = cev_paths(
        F0, sigma, beta, T,
        n_time, n_sim, seed
    )
    F_T = F_paths[:, -1]
    if option_type.lower() == "call":
        payoff = np.maximum(F_T - X, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(X - F_T, 0.0)
    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )
    price = (
        np.exp(-r * T)
        * np.mean(payoff)
    )
    return float(price)


#========================================================
# CEV European Option Valuation: Crank-Nicolson Scheme
#========================================================
def cev_option_price_cn(
    F0, X, r, sigma, beta, T,
    Nt=250, NF=200,
    Fmin=0.0, Fmax=None,
    option_type="call"
):
    if Fmax is None:
        width = (
            6 * sigma
            * max(F0, X)**beta
            * np.sqrt(T)
        )
        Fmax = max(F0, X) + width
    F_grid = np.linspace(Fmin, Fmax, NF)
    dt = T / Nt
    L = matrix_L_cev(
        NF, F_grid, r, sigma, beta
    )
    I = np.eye(NF)
    M1 = I - 0.5 * dt * L
    M2 = I + 0.5 * dt * L
    if option_type.lower() == "call":
        V = np.maximum(F_grid - X, 0.0)
    elif option_type.lower() == "put":
        V = np.maximum(X - F_grid, 0.0)
    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )
    for n in reversed(range(Nt)):
        t = n * dt
        V = solve(M1, M2 @ V)
        tau = T - t
        if option_type.lower() == "call":
            V[0] = 0.0
            V[-1] = np.exp(-r * tau) * (
                F_grid[-1] - X
            )
        else:
            V[0] = np.exp(-r * tau) * X
            V[-1] = 0.0
    price = np.interp(
        F0, F_grid, V
    )
    return float(price)


#===================== CEV European Swaption Valuation =====================


#=======================================================
# CEV European Swaption Valuation: Analytical Method
#=======================================================
def cev_swaption_price_analytical(
    S0, X, r, sigma, beta, T,
    payment_times,
    swaption_type="payer"
):
    payment_times = np.asarray(payment_times)
    if np.any(payment_times <= T):
        raise ValueError(
            "All payment_times must be greater than swaption expiry T"
        )
    if np.any(np.diff(payment_times) <= 0):
        raise ValueError(
            "payment_times must be strictly increasing"
        )
    delta = np.diff(
        np.concatenate(([T], payment_times))
    )
    P = np.exp(-r * payment_times)
    A0 = np.sum(delta * P)
    if swaption_type.lower() == "payer":
        cev_type = "call"
    elif swaption_type.lower() == "receiver":
        cev_type = "put"
    else:
        raise ValueError(
            "swaption_type must be 'payer' or 'receiver'"
        )
    cev_price = cev_option_price_analytical(
        S0, X, 0.0, sigma, beta, T,
        option_type=cev_type
    )
    price = A0 * cev_price
    return float(price)


#==========================================================
# CEV European Swaption Valuation: Monte Carlo Simulation
#==========================================================
def cev_swaption_price_mc(
    S0, X, r, sigma, beta, T,
    payment_times, n_time, n_sim,
    swaption_type="payer", seed=None
):
    payment_times = np.asarray(payment_times)
    if np.any(payment_times <= T):
        raise ValueError(
            "All payment_times must be greater than swaption expiry T"
        )
    if np.any(np.diff(payment_times) <= 0):
        raise ValueError(
            "payment_times must be strictly increasing"
        )
    delta = np.diff(
        np.concatenate(([T], payment_times))
    )
    P = np.exp(-r * payment_times)
    A0 = np.sum(delta * P)
    S_paths = cev_paths(
        S0, sigma, beta, T,
        n_time, n_sim, seed
    )
    S_T = S_paths[:, -1]
    if swaption_type.lower() == "payer":
        payoff = np.maximum(S_T - X, 0.0)
    elif swaption_type.lower() == "receiver":
        payoff = np.maximum(X - S_T, 0.0)
    else:
        raise ValueError(
            "swaption_type must be 'payer' or 'receiver'"
        )
    price = A0 * np.mean(payoff)
    return float(price)


#=========================================================
# CEV European Swaption Valuation: Crank-Nicolson Scheme
#=========================================================
def cev_swaption_price_cn(
    S0, X, r, sigma, beta, T,
    payment_times,
    Nt=250, NS=200,
    Smin=0.0, Smax=None,
    swaption_type="payer"
):
    payment_times = np.asarray(payment_times)
    if np.any(payment_times <= T):
        raise ValueError(
            "All payment_times must be greater than swaption expiry T"
        )
    if np.any(np.diff(payment_times) <= 0):
        raise ValueError(
            "payment_times must be strictly increasing"
        )
    delta = np.diff(
        np.concatenate(([T], payment_times))
    )
    P = np.exp(-r * payment_times)
    A0 = np.sum(delta * P)
    if Smax is None:
        width = (
            6 * sigma
            * max(S0, X)**beta
            * np.sqrt(T)
        )
        Smax = max(S0, X) + width
    S_grid = np.linspace(Smin, Smax, NS)
    dt = T / Nt
    L = matrix_L_cev(
        NS, S_grid, 0.0, sigma, beta
    )
    I = np.eye(NS)
    M1 = I - 0.5 * dt * L
    M2 = I + 0.5 * dt * L
    if swaption_type.lower() == "payer":
        V = np.maximum(S_grid - X, 0.0)
    elif swaption_type.lower() == "receiver":
        V = np.maximum(X - S_grid, 0.0)
    else:
        raise ValueError(
            "swaption_type must be 'payer' or 'receiver'"
        )
    for _ in reversed(range(Nt)):
        V = solve(M1, M2 @ V)
        if swaption_type.lower() == "payer":
            V[0] = 0.0
            V[-1] = S_grid[-1] - X
        else:
            V[0] = X
            V[-1] = 0.0
    price = A0 * np.interp(
        S0, S_grid, V
    )
    return float(price)