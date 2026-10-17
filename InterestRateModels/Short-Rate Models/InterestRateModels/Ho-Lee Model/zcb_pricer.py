import numpy as np

from helpers import matrix_L_ho_lee, ho_lee_lattice


#====================================================
# Zero-Coupon Bond Valuation: Ho-Lee Analytical
#====================================================
def zcb_ho_lee_analytical(r_t, sigma, t, T, P0, f0_t, N=1.0):
    B = T - t
    P0_T = P0(T)
    P0_t = P0(t)
    sigma_term = 0.5 * sigma**2 * t * B**2
    log_A = np.log(P0_T / P0_t) + B * f0_t - sigma_term
    A = np.exp(log_A)
    price = N * A * np.exp(-B * r_t)
    return price


#====================================================
# Zero-Coupon Bond Valuation: Ho-Lee Monte Carlo
#====================================================
def zcb_ho_lee_mc(r_paths, t, T, N=1.0):
    n_time = r_paths.shape[1] - 1
    dt = (T - t) / n_time
    integral_r = np.trapezoid(r_paths, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)
    price = N * np.mean(discount_factors)
    return price


#====================================================
# Zero-Coupon Bond Valuation: Ho-Lee Implicit FDM
#====================================================
def zcb_ho_lee_fdm(r0, theta, sigma, T, Nr=301, Nt=500, r_min=-0.10, r_max=0.20, N=1.0):
    r_grid = np.linspace(r_min, r_max, Nr)
    dt = T / Nt
    V = np.full(Nr, N, dtype=float)
    I = np.eye(Nr)

    for n in range(Nt - 1, -1, -1):
        t = n * dt
        theta_t = theta(t) if callable(theta) else theta
        L = matrix_L_ho_lee(Nr, r_grid, theta_t, sigma)
        A = I - dt * L

        tau = T - t
        V[0] = N * np.exp(-r_grid[0] * tau)
        V[-1] = N * np.exp(-r_grid[-1] * tau)

        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0

        V = np.linalg.solve(A, V)

    price = np.interp(r0, r_grid, V)
    return price


#====================================================
# Zero-Coupon Bond Valuation: Ho-Lee Lattice
#====================================================
def zcb_ho_lee_lattice(sigma, T, n_steps, P0, N=1.0):
    dt = T / n_steps
    r_tree, _, _ = ho_lee_lattice(sigma=sigma, T=T, n_steps=n_steps, P0=P0)
    V = np.full(n_steps + 1, N, dtype=float)

    for n in range(n_steps - 1, -1, -1):
        V_new = np.zeros(n + 1)

        for j in range(n + 1):
            continuation = 0.5 * V[j] + 0.5 * V[j + 1]
            V_new[j] = np.exp(-r_tree[n, j] * dt) * continuation

        V = V_new

    price = V[0]
    return price