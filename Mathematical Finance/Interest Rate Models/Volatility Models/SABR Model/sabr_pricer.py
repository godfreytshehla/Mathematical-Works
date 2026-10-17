import numpy as np
from scipy.stats import norm
from scipy.linalg import solve
from sabr_helpers import *
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from sabr_helpers import *

#================ European Option on Forward Valuation: SABR Model =================


#=======================================================
# SABR Implied Volatility: Analytical Approximation
#=======================================================
def sabr_implied_vol_analytical(
    F0, X, nu0, beta, sigma, rho, T
):
    if F0 <= 0 or X <= 0:
        raise ValueError("F0 and X must be positive")
    if np.isclose(F0, X):
        correction = (
            ((1 - beta)**2 / 24)
            * (nu0**2 / F0**(2 * (1 - beta)))
            + (rho * beta / 4)
            * (sigma * nu0 / F0**(1 - beta))
            + ((2 - 3 * rho**2) / 24)
            * sigma**2
        )
        vol = (
            nu0 * F0**(beta - 1)
            * (1 + correction * T)
        )
        return float(vol)
    Fmid = np.sqrt(F0 * X)
    if np.isclose(beta, 1.0):
        zeta = (
            sigma / nu0
            * np.log(F0 / X)
        )
    else:
        zeta = (
            sigma / (nu0 * (1 - beta))
            * (F0**(1 - beta) - X**(1 - beta))
        )
    D = np.log(
        (
            np.sqrt(1 - 2 * rho * zeta + zeta**2)
            + zeta - rho
        )
        / (1 - rho)
    )
    gamma1 = beta / Fmid
    gamma2 = -beta * (1 - beta) / Fmid**2
    correction = (
        (
            2 * gamma2
            - gamma1**2
            + 1 / Fmid**2
        )
        / 24
        * (
            nu0 * Fmid**beta / sigma
        )**2
        + rho * gamma1 / 4
        * (
            nu0 * Fmid**beta / sigma
        )
        + (2 - 3 * rho**2) / 24
    )
    vol = (
        sigma
        * np.log(F0 / X)
        / D
        * (1 + correction * sigma**2 * T)
    )
    return float(vol)

#=======================================================
# SABR European Option Valuation: Analytical Approximation
#=======================================================
def sabr_option_price_analytical(
    F0, X, r, nu0, beta, sigma, rho, T,
    option_type="call"
):
    sabr_vol = sabr_implied_vol_analytical(
        F0, X, nu0, beta, sigma, rho, T
    )
    d1 = (
        np.log(F0 / X)
        + 0.5 * sabr_vol**2 * T
    ) / (sabr_vol * np.sqrt(T))
    d2 = d1 - sabr_vol * np.sqrt(T)
    if option_type.lower() == "call":
        price = np.exp(-r * T) * (
            F0 * norm.cdf(d1)
            - X * norm.cdf(d2)
        )
    elif option_type.lower() == "put":
        price = np.exp(-r * T) * (
            X * norm.cdf(-d2)
            - F0 * norm.cdf(-d1)
        )
    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )
    return float(price)


#=============================================================
# SABR European Option Valuation: Monte Carlo Simulation
#=============================================================
def sabr_option_price_mc(
    F0, X, r, nu0, beta, sigma, rho, T, n_time, n_sim,
    option_type="call", seed=None
):
    F_paths = sabr_paths(
        F0, nu0, beta, sigma, rho, T, n_time, n_sim, seed
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


#==========================================================
# SABR European Option Valuation: Modified Craig-Sneyd ADI
#==========================================================
def sabr_option_price_mcs(
    F0, X, r, nu0, beta, sigma, rho, T,
    Nt=250, NF=200, Nnu=100,
    Fmin=0.0, Fmax=None, numax=None,
    option_type="call", theta=2/3
):
    if Fmax is None:
        Fmax = 10 * X
    if numax is None:
        numax = nu0 * np.exp(4 * sigma * np.sqrt(T))
    # F = np.linspace(Fmin, Fmax, NF)
    # nu = np.linspace(0.0, numax, Nnu)
    F = S_grid(Fmax, X, NF)
    nu = v_grid(numax, Nnu)
    dt = T / Nt
    m = NF * Nnu
    def idx(i, j):
        return i * Nnu + j
    L0 = matrix_L0(F, nu, beta, rho, sigma)
    L1 = matrix_L1(F, nu, beta, r)
    L2 = matrix_L2(F, nu, sigma, r)
    L = L0 + L1 + L2
    I = identity(m)
    ML1 = (I - theta * dt * L1).tocsr()
    ML2 = (I - theta * dt * L2).tocsr()
    if option_type.lower() == "call":
        payoff = np.maximum(F - X, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(X - F, 0.0)
    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )
    U = np.repeat(payoff, Nnu)
    for n in range(Nt):
        tau = (n + 1) * dt
        Y0 = U + dt * (L @ U)
        Y1_rhs = Y0 - theta * dt * (L1 @ U)
        Y1 = spsolve(ML1, Y1_rhs)
        Y2_rhs = Y1 - theta * dt * (L2 @ U)
        Y2 = spsolve(ML2, Y2_rhs)
        Y3 = Y0 + theta * dt * L0 @ (Y2 - U)
        Y4 = Y3 + (0.5 - theta) * dt * L @ (Y2 - U)
        Y5_rhs = Y4 - theta * dt * (L1 @ U)
        Y5 = spsolve(ML1, Y5_rhs)
        Y6_rhs = Y5 - theta * dt * (L2 @ U)
        U = spsolve(ML2, Y6_rhs)
        for i in range(NF):
            U[idx(i, 0)] = np.exp(-r * tau) * payoff[i]
            U[idx(i, Nnu - 1)] = U[idx(i, Nnu - 2)]
        for j in range(Nnu):
            if option_type.lower() == "call":
                U[idx(0, j)] = 0.0
                U[idx(NF - 1, j)] = (
                    np.exp(-r * tau) * (Fmax - X)
                )
            else:
                U[idx(0, j)] = (
                    np.exp(-r * tau) * (X - Fmin)
                )
                U[idx(NF - 1, j)] = 0.0
    U_grid = U.reshape(NF, Nnu)
    interp = RegularGridInterpolator((F, nu), U_grid)
    price = interp([[F0, nu0]])[0]
    return float(price)


#===================== Bachelier European Swaption Valuation =====================


#=========================================================
# SABR European Swaption Valuation: Monte Carlo Simulation
#=========================================================
def sabr_swaption_price_mc(
    S0, X, r, nu0, beta, sigma, rho, T,
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
    S_paths = sabr_paths(
        S0, nu0, beta, sigma, rho,
        T, n_time, n_sim, seed
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


#============================================================
# SABR European Swaption Valuation: Modified Craig-Sneyd ADI
#============================================================
def sabr_swaption_price_mcs(
    S0, X, r, nu0, beta, sigma, rho, T,
    payment_times,
    Nt=250, NS=200, Nnu=100,
    Smin=0.0, Smax=None, numax=None,
    swaption_type="payer", theta=2/3
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
        Smax = 10 * X
    if numax is None:
        numax = nu0 * np.exp(4 * sigma * np.sqrt(T))
    S = S_grid(Smax, X, NS)
    nu = v_grid(numax, Nnu)
    dt = T / Nt
    m = NS * Nnu
    def idx(i, j):
        return i * Nnu + j
    L0 = matrix_L0(S, nu, beta, rho, sigma)
    L1 = matrix_L1(S, nu, beta, 0.0)
    L2 = matrix_L2(S, nu, sigma, 0.0)
    L = L0 + L1 + L2
    I = identity(m)
    ML1 = (I - theta * dt * L1).tocsr()
    ML2 = (I - theta * dt * L2).tocsr()
    if swaption_type.lower() == "payer":
        payoff = np.maximum(S - X, 0.0)
    elif swaption_type.lower() == "receiver":
        payoff = np.maximum(X - S, 0.0)
    else:
        raise ValueError(
            "swaption_type must be 'payer' or 'receiver'"
        )
    U = np.repeat(payoff, Nnu)
    for _ in range(Nt):
        Y0 = U + dt * (L @ U)
        Y1_rhs = Y0 - theta * dt * (L1 @ U)
        Y1 = spsolve(ML1, Y1_rhs)
        Y2_rhs = Y1 - theta * dt * (L2 @ U)
        Y2 = spsolve(ML2, Y2_rhs)
        Y3 = Y0 + theta * dt * L0 @ (Y2 - U)
        Y4 = Y3 + (0.5 - theta) * dt * L @ (Y2 - U)
        Y5_rhs = Y4 - theta * dt * (L1 @ U)
        Y5 = spsolve(ML1, Y5_rhs)
        Y6_rhs = Y5 - theta * dt * (L2 @ U)
        U = spsolve(ML2, Y6_rhs)
        for i in range(NS):
            U[idx(i, 0)] = payoff[i]
            U[idx(i, Nnu - 1)] = U[idx(i, Nnu - 2)]
        for j in range(Nnu):
            if swaption_type.lower() == "payer":
                U[idx(0, j)] = 0.0
                U[idx(NS - 1, j)] = Smax - X
            else:
                U[idx(0, j)] = X - Smin
                U[idx(NS - 1, j)] = 0.0
    U_grid = U.reshape(NS, Nnu)
    interp = RegularGridInterpolator((S, nu), U_grid)
    price = A0 * interp([[S0, nu0]])[0]
    return float(price)


#=========================================================
# SABR European Swaption Valuation: Analytical Approximation
#=========================================================
def sabr_swaption_price_analytical(
    S0, X, r, nu0, beta, sigma, rho, T,
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
    sabr_vol = sabr_implied_vol_analytical(
        S0, X, nu0, beta, sigma, rho, T
    )
    d1 = (
        np.log(S0 / X)
        + 0.5 * sabr_vol**2 * T
    ) / (sabr_vol * np.sqrt(T))
    d2 = d1 - sabr_vol * np.sqrt(T)
    if swaption_type.lower() == "payer":
        price = A0 * (
            S0 * norm.cdf(d1)
            - X * norm.cdf(d2)
        )
    elif swaption_type.lower() == "receiver":
        price = A0 * (
            X * norm.cdf(-d2)
            - S0 * norm.cdf(-d1)
        )
    else:
        raise ValueError(
            "swaption_type must be 'payer' or 'receiver'"
        )
    return float(price)