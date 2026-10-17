import numpy as np

#===========================================
# Discretized Differential Operator (CEV)
#===========================================
def matrix_L_cev(NF, F_grid, r, sigma, beta):
    dF = F_grid[1] - F_grid[0]
    def L_disc():
        return -r * np.eye(NF)
    def L_FF():
        L = np.zeros((NF, NF))
        for i in range(1, NF - 1):
            coeff = (
                0.5
                * sigma**2
                * F_grid[i]**(beta)
                / dF**2
            )
            L[i, i - 1] = coeff
            L[i, i] = -2 * coeff
            L[i, i + 1] = coeff
        return L
    return L_disc() + L_FF()


#=====================================================
# CEV model path generation using discretization
#=====================================================
def cev_paths(F0, sigma, beta, T, n_time, n_sim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_time
    F = np.zeros((n_sim, n_time + 1))
    F[:, 0] = F0
    for t in range(1, n_time + 1):
        z = np.random.normal(size=n_sim)
        F[:, t] = np.maximum(
            F[:, t - 1]
            + sigma
            * np.maximum(F[:, t - 1], 0.0)**(beta/2)
            * np.sqrt(dt) * z,
            0.0
        )
    return F