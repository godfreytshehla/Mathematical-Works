from dataclasses import dataclass
import numpy as np

#============================================
# Option Parameters
#============================================
@dataclass
class OptionParams:
    S0: float
    X: float
    T: float
    r: float
    q: float
    sigma: float

#============================================
# Geometric Brownian Motion Paths (BSM)
#============================================
def gbm_paths(p, n_time, n_sim, seed=None):
    rng = np.random.default_rng(seed)
    dt = p.T / n_time
    Z = rng.standard_normal((n_time, n_sim))
    increments = (p.r - p.q - 0.5 * p.sigma**2) * dt + p.sigma * np.sqrt(dt) * Z
    S = np.empty((n_time + 1, n_sim))
    S[0] = p.S0
    S[1:] = p.S0 * np.exp(np.cumsum(increments, axis=0))
    return S

#============================================
# Discretized Differential Operator (BSM)
#============================================
def matrix_L_bsm(p, S_grid):
    NS = len(S_grid)
    dS = S_grid[1] - S_grid[0]
    def L_disc():
        return -p.r * np.eye(NS)
    def L_S():
        L = np.zeros((NS, NS))
        for i in range(1, NS - 1):
            coeff = (p.r - p.q) * S_grid[i] / (2 * dS)
            L[i, i - 1] = -coeff
            L[i, i + 1] = coeff
        return L
    def L_SS():
        L = np.zeros((NS, NS))
        for i in range(1, NS - 1):
            coeff = 0.5 * p.sigma**2 * S_grid[i]**2 / dS**2
            L[i, i - 1] = coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] = coeff
        return L
    return L_disc() + L_S() + L_SS()