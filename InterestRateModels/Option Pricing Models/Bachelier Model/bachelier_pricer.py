import numpy as np
from scipy.stats import norm
from scipy.linalg import solve
from bachelier_helpers import *


#================ European Option on Forward Valuation: Bachelier Model =================


#==========================================================
# Bachelier European Option Valuation: Analytical Method
#==========================================================
def bachelier_option_price_analytical(
    F0, X, r, sigma, T,
    option_type="call"
):
    d = (F0 - X) / (sigma * np.sqrt(T))

    if option_type.lower() == "call":
        price = np.exp(-r * T) * (
            (F0 - X) * norm.cdf(d)
            + sigma * np.sqrt(T) * norm.pdf(d)
        )

    elif option_type.lower() == "put":
        price = np.exp(-r * T) * (
            (X - F0) * norm.cdf(-d)
            + sigma * np.sqrt(T) * norm.pdf(d)
        )

    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )

    return float(price)


#=============================================================
# Bachelier European Option Valuation: Monte Carlo Simulation
#=============================================================
def bachelier_option_price_mc(
    F0, X, r, sigma, T, n_time, n_sim,
    option_type="call", seed=None
):
    F_paths = bachelier_paths(
        F0, sigma, T, n_time, n_sim, seed
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

    disc = np.exp(-r * T)

    return float(disc * np.mean(payoff))


#=============================================================
# Bachelier European Option Valuation: Crank-Nicolson Scheme
#=============================================================
def bachelier_option_price_cn(
    F0, X, r, sigma, T,
    Nt=250, NF=200,
    Fmin=None, Fmax=None,
    option_type="call"
):
    width = 6 * sigma * np.sqrt(T)

    if Fmin is None:
        Fmin = min(F0, X) - width

    if Fmax is None:
        Fmax = max(F0, X) + width

    F_grid = np.linspace(Fmin, Fmax, NF)
    dt = T / Nt

    L = matrix_L_bachelier(
        NF, F_grid, r, sigma
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
            V[0] = np.exp(-r * tau) * (
                X - F_grid[0]
            )
            V[-1] = 0.0

    price = np.interp(F0, F_grid, V)

    return float(price)


#===================== Bachelier European Swaption Valuation =====================


#==========================================================
# Bachelier European Swaption Valuation: Analytical Method
#==========================================================
def bachelier_swaption_price_analytical(
    S0, X, r, sigma, T,
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
        bachelier_type = "call"

    elif swaption_type.lower() == "receiver":
        bachelier_type = "put"

    else:
        raise ValueError(
            "swaption_type must be 'payer' or 'receiver'"
        )

    bachelier_price = bachelier_option_price_analytical(
        S0, X, r, sigma, T,
        option_type=bachelier_type
    )

    price = A0 * np.exp(r * T) * bachelier_price

    return float(price)


#==============================================================
# Bachelier European Swaption Valuation: Monte Carlo Simulation
#==============================================================
def bachelier_swaption_price_mc(
    S0, X, r, sigma, T,
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

    S_paths = bachelier_paths(
        S0, sigma, T, n_time, n_sim, seed
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


#=============================================================
# Bachelier European Swaption Valuation: Crank-Nicolson Scheme
#=============================================================
def bachelier_swaption_price_cn(
    S0, X, r, sigma, T,
    payment_times,
    Nt=250, NS=200,
    Smin=None, Smax=None,
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

    width = 6 * sigma * np.sqrt(T)

    if Smin is None:
        Smin = min(S0, X) - width

    if Smax is None:
        Smax = max(S0, X) + width

    S_grid = np.linspace(Smin, Smax, NS)
    dt = T / Nt

    L = matrix_L_bachelier(
        NS, S_grid, 0.0, sigma
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

    for n in reversed(range(Nt)):
        V = solve(M1, M2 @ V)

        if swaption_type.lower() == "payer":
            V[0] = 0.0
            V[-1] = S_grid[-1] - X

        else:
            V[0] = X - S_grid[0]
            V[-1] = 0.0

    price = A0 * np.interp(
        S0, S_grid, V
    )

    return float(price)