import numpy as np

#============================================
# Discretized Differential Operator (Vasicek)
#============================================
def matrix_L_vasicek(Nr, r_grid, a, b, sigma):
    dr = r_grid[1] - r_grid[0]
    # -----------------------------
    # discounting term: -r f
    # -----------------------------
    def L_disc():
        return -np.diag(r_grid)
    # -----------------------------
    # drift term: a(b - r) f_r
    # central difference
    # -----------------------------
    def L_r():
        L = np.zeros((Nr, Nr))
        for i in range(1, Nr - 1):
            drift = a * (b - r_grid[i])
            L[i, i - 1] = -drift / (2 * dr)
            L[i, i + 1] =  drift / (2 * dr)
        return L
    # -----------------------------
    # diffusion term: (sigma^2 / 2) f_rr
    # constant coefficient
    # -----------------------------
    def L_rr():
        L = np.zeros((Nr, Nr))
        coeff = 0.5 * sigma**2 / dr**2
        for i in range(1, Nr - 1):
            L[i, i - 1] = coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] = coeff
        return L
    return L_disc() + L_r() + L_rr()


#=========================================================
# Vasicek model path generation using exact discretization
#=========================================================

def vasicek_paths(r0, a, b, sigma, T, n_time, n_sim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_time
    r = np.zeros((n_sim, n_time + 1))
    r[:, 0] = r0
    mean_reversion = np.exp(-a * dt)
    for t in range(1, n_time + 1):
        # exact conditional mean
        drift = b * (1 - mean_reversion)
        # exact conditional variance
        variance = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * dt))
        z = np.random.normal(size=n_sim)
        r[:, t] = (
            r[:, t - 1] * mean_reversion
            + drift
            + np.sqrt(variance) * z
        )
    return r