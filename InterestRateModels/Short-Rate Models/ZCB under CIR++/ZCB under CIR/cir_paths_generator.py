import numpy as np

#===============================================
# CIR short rate simulation
#===============================================
def cir_paths(r0, a, b, sigma, T, n_time, n_sim):
    """
    CIR short rate simulation with Feller's condition check.
    """
    # Check Feller's condition
    if 2 * a * b < sigma**2:
        raise ValueError(f"Feller's condition violated: 2*a*b = {2*a*b} < sigma^2 = {sigma**2}")
    dt = T / n_time
    r = np.zeros((n_sim, n_time+1))
    r[:, 0] = r0
    dW = np.random.normal(0.0, np.sqrt(dt), size=(n_sim, n_time))
    for t in range(1, n_time+1):
        r_prev = np.maximum(r[:, t-1], 0)
        dr = a * (b - r_prev) * dt + sigma * np.sqrt(r_prev) * dW[:, t-1]
        r[:, t] = r_prev + dr
        # Optional: full truncation (ensures non-negativity)
        r[:, t] = np.maximum(r[:, t], 0)
    return r
