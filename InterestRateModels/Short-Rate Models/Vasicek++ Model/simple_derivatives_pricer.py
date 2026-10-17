import numpy as np
from scipy.linalg import solve
from vasicek_helpers import*

#=========================================================
# Zero-Coupon Bond Valuation: Closed-Form Solution
#=========================================================
def vasicek_zcb_price(r0, a, b, sigma, T):
        # B(t, T)
        Bt = (1 - np.exp(-a * T)) / a
        # A(t, T)
        term1 = (
            b - sigma**2 / (2 * a**2)
        ) * (Bt - T)
        term2 = (
            sigma**2 / (4 * a)
        ) * Bt**2
        At = np.exp(term1 - term2)
        # ZCB price
        return At * np.exp(-Bt * r0)

#=====================================================
# Zero-Coupon Bond Valuation: Monte Carlo Simulation
#=====================================================
def zcb_price_mc_vasicek(r0, a, b, sigma, T, n_time, n_sim, seed=None):
    dt = T / n_time
    r_paths = vasicek_paths(r0, a, b, sigma, T, n_time, n_sim, seed)
    disc = np.exp(-np.sum(r_paths[:, :-1] * dt, axis=1))
    return np.mean(disc)

#====================================================
# Zero-Coupon Bond Valuation: Crank-Nicolson Scheme
#====================================================
def zcb_price_cn_vasicek(r0, a, b, sigma, T, Nt=250, Nr=200, N=1.0, rmax=None):
    if rmax is None:
        rmax = b + 5 * sigma + 3 * r0
    r_grid = np.linspace(0, rmax, Nr)
    dr = r_grid[1] - r_grid[0]
    dt = T / Nt
    L = matrix_L_vasicek(Nr, r_grid, a, b, sigma)
    I = np.eye(Nr)
    M1 = I - 0.5 * dt * L
    M2 = I + 0.5 * dt * L
    V = np.full(Nr, N)
    for n in reversed(range(Nt)):
        t = n * dt
        V = solve(M1, M2 @ V)
        V[0] = N * np.exp(-r_grid[0] * (T - t))  # r -> 0 boundary
        V[-1] = 0                                 # r -> r_max boundary
    return np.interp(r0, r_grid, V)

#============================================================
# Forward Rate Agreement (FRA) Valuation under Vasicek Model
#============================================================
def fra_price_vasicek(r0, a, b, sigma, T1, T2, K, N=1.0):
    if T1 >= T2:
        raise ValueError("Require T1 < T2 for FRA pricing")
    P1 = vasicek_zcb_price(r0, a, b, sigma, T1)
    P2 = vasicek_zcb_price(r0, a, b, sigma, T2)
    dt = T2 - T1
    forward_rate = (P1 / P2 - 1) / dt
    price = N * dt * P2 * (forward_rate - K)
    return price


#========================================================
# Interest Rate Swap (IRS) Valuation under Vasicek Model
#========================================================
def irs_price_vasicek(r0, a, b, sigma, Ts, K, N=1.0):
    Ts = np.asarray(Ts)
    P = np.array([
        vasicek_zcb_price(r0, a, b, sigma, T)
        for T in Ts
    ])
    dt = np.diff(Ts)
    fixed_leg = N * K * np.sum(dt * P[1:])
    floating_leg = N * (1 - P[-1])
    return fixed_leg - floating_leg


#===============================================
# Par Swap Rate Calculation under Vasicek Model
#===============================================
def par_swap_rate_vasicek(r0, a, b, sigma, Ts, N=1.0):
    Ts = np.asarray(Ts)
    if len(Ts) < 2:
        raise ValueError("Need at least 2 swap dates")
    P = np.array([
        vasicek_zcb_price(r0, a, b, sigma, T)
        for T in Ts
    ])
    dt = np.diff(Ts)
    numerator = 1 - P[-1]
    denominator = np.sum(dt * P[1:])
    return numerator / denominator