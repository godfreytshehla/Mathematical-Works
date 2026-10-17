import numpy as np

#===================================================================
# Hull-White model path generation using exact solution
#===================================================================
def hw_paths_exact(r0, a, theta, sigma, T, n_time, n_sim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_time
    r = np.zeros((n_sim, n_time + 1))
    r[:, 0] = r0
    for t in range(1, n_time + 1):
        time = (t - 1) * dt
        theta_t = theta(time) if callable(theta) else theta
        mean_reversion = np.exp(-a * dt)
        # drift integral approximation (constant theta case shown)
        drift = theta_t * (1 - mean_reversion) / a
        variance = (sigma**2 / (2*a)) * (1 - np.exp(-2*a*dt))
        r[:, t] = (
            r[:, t - 1] * mean_reversion
            + drift
            + np.sqrt(variance) * np.random.normal(size=n_sim)
        )
    return r