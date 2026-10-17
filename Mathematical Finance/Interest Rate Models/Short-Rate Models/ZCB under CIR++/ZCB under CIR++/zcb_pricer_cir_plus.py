import numpy as np
from cir_paths_generator import*
from fdm_matrices import*
from scipy.linalg import solve

#====================================================
# Zero-Coupon Bond Valuation: Analytical Solution
#====================================================
def zcb_cir_analytical(r0, a, b, sigma, T, N=1.0):
    """
    r0, a, b, sigma : CIR parameters
    T               : maturity
    N               : face value of the bond
    """
    h = np.sqrt(a**2 + 2 * sigma**2)
    # B(0, T)
    numerator_B = 2 * (np.exp(h * T) - 1)
    denominator_B = 2 * h + (a + h) * (np.exp(h * T) - 1)
    B = numerator_B / denominator_B
    # A(0, T)
    numerator_A = 2 * h * np.exp((a + h) * T / 2)
    denominator_A = 2 * h + (a + h) * (np.exp(h * T) - 1)
    A = (numerator_A / denominator_A) ** (2 * a * b / sigma**2)
    # Zero-coupon bond price
    return N * A * np.exp(-B * r0)

#====================================================
# Zero-Coupon Bond Valuation: Monte Carlo Simulation
#====================================================
def zcb_cir_mc(r0, a, b, sigma, T, n_time, n_sim, N=1.0):
    """
    r0, a, b, sigma : CIR parameters
    T               : maturity
    n_time, n_sim          : time steps, number of paths
    N               : face value of the bond
    """
    # simulate short rate paths and compute time step
    r, dt = cir_paths(r0, a, b, sigma, T, n_time, n_sim), T / n_time
    # pathwise discount factor from 0 to T
    discount_factor = np.exp(-np.sum(r[:, :-1] * dt, axis=1))
    discounted_payoff = N * discount_factor
    return np.mean(discounted_payoff)

#====================================================
# Zero-Coupon Bond Valuation: Crank-Nicolson Scheme
#====================================================
def zcb_cir_cn(r0, a, b, sigma, T, Nt=150, Nr=200, N=1.0):
    rmax = b + 5 * sigma + 3 * r0 # maybe this should be an input?
    r_grid = np.linspace(0, rmax, Nr)
    dr = r_grid[1] - r_grid[0]
    dt = T / Nt
    L = matrix_L(Nr, r_grid, a, b, sigma)
    I = np.eye(Nr)
    M1, M2 = I - 0.5 * dt * L, I + 0.5 * dt * L
    V = np.full(Nr, N)
    for n in reversed(range(Nt)):
        t = n * dt
        V = solve(M1, M2 @ V)
        V[0] = N * np.exp(-r_grid[0] * (T - t))   # r -> 0
        V[-1] = 0                                 # r -> r_max
    return np.interp(r0, r_grid, V)
