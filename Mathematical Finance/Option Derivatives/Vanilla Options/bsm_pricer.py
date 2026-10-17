from bsm_helpers import gbm_paths, matrix_L_bsm
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve
from scipy.integrate import quad

@dataclass
class OptionParams:
    S0: float
    X: float
    T: float
    r: float
    q: float
    sigma: float

def analytical_bsm_option_price(p, option_type="call"):
    d1 = (np.log(p.S0 / p.X) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    if option_type == "call":
        return p.S0 * np.exp(-p.q * p.T) * norm.cdf(d1) - p.X * np.exp(-p.r * p.T) * norm.cdf(d2)
    elif option_type == "put":
        return p.X * np.exp(-p.r * p.T) * norm.cdf(-d2) - p.S0 * np.exp(-p.q * p.T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def monte_carlo_option_price(p, n_time, n_sim, option_type="call", seed=None):
    paths = gbm_paths(p, n_time, n_sim, seed)
    ST = paths[-1]
    if option_type == "call":
        payoff = np.maximum(ST - p.X, 0)
    elif option_type == "put":
        payoff = np.maximum(p.X - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return np.exp(-p.r * p.T) * np.mean(payoff)

def binomial_option_price(p, n_t, option_type="call"):
    dt = p.T / n_t
    u = np.exp(p.sigma * np.sqrt(dt))
    d = 1 / u
    prob = (np.exp((p.r - p.q) * dt) - d) / (u - d)
    j = np.arange(n_t + 1)
    ST = p.S0 * u**j * d**(n_t - j)
    if option_type == "call":
        V = np.maximum(ST - p.X, 0)
    elif option_type == "put":
        V = np.maximum(p.X - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    df = np.exp(-p.r * dt)
    for i in range(n_t - 1, -1, -1):
        V = df * (prob * V[1:i + 2] + (1 - prob) * V[:i + 1])
    return V[0]

def bsm_option_price_fdm(p, N_t=250, N_s=200, Smax=None, option_type="call", fdm_type="crank_nicolson", theta=None):
    if Smax is None:
        Smax = 4 * max(p.S0, p.X)
    if fdm_type == "explicit_method":
        theta = 0.0
    elif fdm_type == "implicit_method":
        theta = 1.0
    elif fdm_type == "crank_nicolson":
        theta = 0.5
    elif fdm_type == "theta_imex":
        if theta is None:
            raise ValueError("theta must be specified for theta_imex")
    else:
        raise ValueError(
            "fdm_type must be 'explicit_method', 'implicit_method', "
            "'crank_nicolson', or 'theta_imex'"
        )
    S_grid = np.linspace(0, Smax, N_s)
    dt = p.T / N_t
    L = matrix_L_bsm(p, S_grid)
    I = np.eye(N_s)
    M1 = I - theta * dt * L
    M2 = I + (1 - theta) * dt * L
    if option_type.lower() == "call":
        V = np.maximum(S_grid - p.X, 0.0)
    elif option_type.lower() == "put":
        V = np.maximum(p.X - S_grid, 0.0)
    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )
    for n in reversed(range(N_t)):
        t = n * dt
        if theta == 0:
            V = M2 @ V
        else:
            V = solve(M1, M2 @ V)
        tau = p.T - t
        if option_type.lower() == "call":
            V[0] = 0.0
            V[-1] = (
                S_grid[-1] * np.exp(-p.q * tau)
                - p.X * np.exp(-p.r * tau)
            )
        else:
            V[0] = p.X * np.exp(-p.r * tau)
            V[-1] = 0.0
    return np.interp(p.S0, S_grid, V)

def bsm_option_price_integration(p, option_type="call"):
    def S_T(z):
        return p.S0 * np.exp(
            (p.r - p.q - 0.5 * p.sigma**2) * p.T
            + p.sigma * np.sqrt(p.T) * z
        )
    def normal_pdf(z):
        return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    z_star = (
        np.log(p.X / p.S0)
        - (p.r - p.q - 0.5 * p.sigma**2) * p.T
    ) / (p.sigma * np.sqrt(p.T))
    if option_type.lower() == "call":
        def integrand(z):
            return (S_T(z) - p.X) * normal_pdf(z)
        value, _ = quad(integrand, z_star, np.inf)
    elif option_type.lower() == "put":
        def integrand(z):
            return (p.X - S_T(z)) * normal_pdf(z)
        value, _ = quad(integrand, -np.inf, z_star)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return np.exp(-p.r * p.T) * value