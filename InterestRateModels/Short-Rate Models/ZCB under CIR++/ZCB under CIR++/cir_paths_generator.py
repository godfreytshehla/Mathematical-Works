import numpy as np

#===============================================
# CIR short rate simulation
#===============================================
def cir_paths(x0, a, b, sigma, T, n_time, n_sim):
    """
    CIR short rate simulation with Feller's condition check.
    """
    # Check Feller's condition
    if 2 * a * b < sigma**2:
        raise ValueError(f"Feller's condition violated: 2*a*b = {2*a*b} < sigma^2 = {sigma**2}")
    dt = T / n_time
    x = np.zeros((n_sim, n_time+1))
    x[:, 0] = x0
    dW = np.random.normal(0.0, np.sqrt(dt), size=(n_sim, n_time))
    for t in range(1, n_time+1):
        x_prev = np.maximum(x[:, t-1], 0)
        dx = a * (b - x_prev) * dt + sigma * np.sqrt(x_prev) * dW[:, t-1]
        x[:, t] = x_prev + dx
        # Optional: full truncation (ensures non-negativity)
        x[:, t] = np.maximum(x[:, t], 0)
    return x
