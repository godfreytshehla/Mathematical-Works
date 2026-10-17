import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

def monte_carlo_capped_option(S0, K, C, r, sigma, T, n_time, n_sim, option_type="call"):
    dt = T / n_time  
    discount_factor = np.exp(-r * T)
    w = np.random.randn(n_sim, n_time)  
    S = np.zeros((n_sim, n_time+1))
    S[:, 0] = S0
    for t in range(1, n_time+1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * w[:, t-1])
    S_T = S[:, -1]
    if option_type == "call":
        payoffs = np.minimum(np.maximum(S_T - K, 0), C)
    elif option_type == "put":
        payoffs = np.minimum(np.maximum(K - S_T, 0), C)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    P0 = discount_factor * np.mean(payoffs)
    return P0

def implicit_capped_option(S0, K, C, r, sigma, T, Ntime, Nstock, option_type="call"):
    dt = T / Ntime  
    Smax = 2 * K 
    S = np.linspace(0, Smax, Nstock + 1)  
    dS = S[1] - S[0]  
    P = np.zeros((Ntime + 1, Nstock + 1))
    if option_type == "call":
        P[-1, :] = np.minimum(C, np.maximum(S - K, 0))  
    elif option_type == "put":
        P[-1, :] = np.minimum(C, np.maximum(K - S, 0))  
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    if option_type == "call":
        P[:, 0] = 0  
        P[:, -1] = C  
    elif option_type == "put":
        P[:, 0] = np.minimum(K, C)  
        P[:, -1] = 0  
    Si = S[1:-1]  
    alpha = 0.5 * dt * (sigma**2 * Si**2 / dS**2 - r * Si / dS)
    beta = -dt * (sigma**2 * Si**2 / dS**2 + r)
    gamma = 0.5 * dt * (sigma**2 * Si**2 / dS**2 + r * Si / dS)
    A = np.diag(1 - beta) + np.diag(-alpha[1:], k=-1) + np.diag(-gamma[:-1], k=1)
    for n in range(Ntime - 1, -1, -1):
        P_n = P[n + 1, 1:-1]  
        P[n, 1:-1] = np.linalg.solve(A, P_n)  
    interpolator = interp1d(S, P[0, :], kind='linear', fill_value="extrapolate")
    return interpolator(S0).item()

def bsm_capped_option_price(S0, K, C, r, sigma, T, option_type="capped_call"):
    def bsm_call(S, X, r, sigma, T):
        d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    def bsm_put(S, X, r, sigma, T):
        d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    if option_type == "call":
        return bsm_call(S0, K, r, sigma, T)
    elif option_type == "put":
        return bsm_put(S0, K, r, sigma, T)
    elif option_type == "capped_call":
        if C is None:
            raise ValueError("Cap (C) must be provided for capped_call option.")
        call_K = bsm_call(S0, K, r, sigma, T)
        call_K_plus_C = bsm_call(S0, K + C, r, sigma, T)
        return call_K - call_K_plus_C
    elif option_type == "capped_put":
        if C is None:
            raise ValueError("Cap (C) must be provided for capped_put option.")
        put_K = bsm_put(S0, K, r, sigma, T)
        put_K_minus_C = bsm_put(S0, K - C, r, sigma, T)
        return put_K - put_K_minus_C
    else:
        raise ValueError("Invalid option_type. Use 'call', 'put', 'capped_call', or 'capped_put'.")