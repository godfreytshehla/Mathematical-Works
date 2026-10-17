import numpy as np

#===============================================
# Discretized Differential Operator (Bachelier)
#===============================================
def matrix_L_bachelier(NF, F_grid, r, sigma):
    dF = F_grid[1] - F_grid[0]
    # -----------------------------
    # discounting term: -r f
    # -----------------------------
    def L_disc():
        return -r * np.eye(NF)
    # -----------------------------
    # diffusion term:
    # (sigma^2 / 2) f_FF
    # constant coefficient
    # -----------------------------
    def L_FF():
        L = np.zeros((NF, NF))
        coeff = (
            0.5
            * sigma**2
            / dF**2
        )
        for i in range(1, NF - 1):
            L[i, i - 1] = coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] = coeff
        return L
    return L_disc() + L_FF()


#=============================================================
# Bachelier model path generation using exact discretization
#=============================================================
def bachelier_paths(F0, sigma, T, n_time, n_sim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_time
    F = np.zeros((n_sim, n_time + 1))
    F[:, 0] = F0
    for t in range(1, n_time + 1):
        z = np.random.normal(size=n_sim)
        F[:, t] = (
            F[:, t - 1]
            + sigma * np.sqrt(dt) * z
        )
    return F