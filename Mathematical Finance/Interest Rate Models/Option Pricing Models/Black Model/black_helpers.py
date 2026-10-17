import numpy as np

#============================================
# Discretized Differential Operator (Black)
#============================================
def matrix_L_black(NF, F_grid, r, sigma):
    dF = F_grid[1] - F_grid[0]
    # -----------------------------
    # discounting term: -r f
    # -----------------------------
    def L_disc():
        return -r * np.eye(NF)
    # -----------------------------
    # diffusion term:
    # (sigma^2 / 2) F^2 f_FF
    # state-dependent coefficient
    # -----------------------------
    def L_FF():
        L = np.zeros((NF, NF))
        for i in range(1, NF - 1):
            coeff = (
                0.5
                * sigma**2
                * F_grid[i]**2
                / dF**2
            )
            L[i, i - 1] = coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] = coeff
        return L
    return L_disc() + L_FF()

#=========================================================
# Black model path generation using exact discretization
#=========================================================

def black_paths(F0, sigma, T, n_time, n_sim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_time
    F = np.zeros((n_sim, n_time + 1))
    F[:, 0] = F0
    for t in range(1, n_time + 1):
        z = np.random.normal(size=n_sim)

        F[:, t] = (
            F[:, t - 1]
            * np.exp(
                -0.5 * sigma**2 * dt
                + sigma * np.sqrt(dt) * z
            )
        )
    return F