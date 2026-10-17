import numpy as np
import matplotlib.pyplot as plt


#====================================================
# Market Zero-Coupon Bond Price
#====================================================
def market_zcb_price(T, P0, N=1.0):
    price = N * P0(T)
    return price


#====================================================
# Ho-Lee Drift Calibration
#====================================================
def ho_lee_theta(t, sigma, forward_rate, h=1e-4):
    if t <= h:
        dfdt = (forward_rate(t + h) - forward_rate(t)) / h
    else:
        dfdt = (forward_rate(t + h) - forward_rate(t - h)) / (2 * h)

    theta = dfdt + sigma**2 * t
    return theta


#====================================================
# Ho-Lee Path Generation
#====================================================
def ho_lee_paths(r0, theta, sigma, T, n_time, n_sim, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_time
    r = np.zeros((n_sim, n_time + 1))
    r[:, 0] = r0

    for n in range(1, n_time + 1):
        time = (n - 1) * dt
        theta_t = theta(time) if callable(theta) else theta
        drift = theta_t * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(size=n_sim)
        r[:, n] = r[:, n - 1] + drift + diffusion

    return r


#====================================================
# Ho-Lee Differential Operator
#====================================================
def matrix_L_ho_lee(Nr, r_grid, theta_t, sigma):
    dr = r_grid[1] - r_grid[0]

    def L_disc():
        return -np.diag(r_grid)

    def L_r():
        L = np.zeros((Nr, Nr))

        for i in range(1, Nr - 1):
            L[i, i - 1] = -theta_t / (2 * dr)
            L[i, i + 1] = theta_t / (2 * dr)

        return L

    def L_rr():
        L = np.zeros((Nr, Nr))
        coeff = 0.5 * sigma**2 / dr**2

        for i in range(1, Nr - 1):
            L[i, i - 1] = coeff
            L[i, i] = -2 * coeff
            L[i, i + 1] = coeff

        return L

    return L_disc() + L_r() + L_rr()


#====================================================
# Ho-Lee Calibrated Short-Rate Lattice
#====================================================
def ho_lee_lattice(sigma, T, n_steps, P0):
    dt = T / n_steps
    h = sigma * np.sqrt(dt)

    r_tree = np.full((n_steps + 1, n_steps + 1), np.nan)
    Q = np.zeros((n_steps + 1, n_steps + 1))
    alpha = np.zeros(n_steps + 1)

    Q[0, 0] = 1.0

    # ------------------------------------------------
    # Calibrate discounting nodes: t_0,...,t_{N-1}
    # ------------------------------------------------
    for n in range(n_steps):
        P_market = P0((n + 1) * dt)

        j = np.arange(n + 1)
        shifts = (2 * j - n) * h

        weighted_sum = np.sum(Q[n, :n + 1] * np.exp(-shifts * dt))
        alpha[n] = np.log(weighted_sum / P_market) / dt

        r_tree[n, :n + 1] = alpha[n] + shifts

        discount = np.exp(-r_tree[n, :n + 1] * dt)

        for j in range(n + 1):
            q = 0.5 * Q[n, j] * discount[j]
            Q[n + 1, j] += q
            Q[n + 1, j + 1] += q

    # ------------------------------------------------
    # Terminal short-rate nodes at T
    # ------------------------------------------------
    if n_steps > 1:
        alpha[n_steps] = alpha[n_steps - 1] + (alpha[n_steps - 1] - alpha[n_steps - 2])
    else:
        alpha[n_steps] = alpha[n_steps - 1]

    j = np.arange(n_steps + 1)
    shifts = (2 * j - n_steps) * h
    r_tree[n_steps, :n_steps + 1] = alpha[n_steps] + shifts

    return r_tree, alpha, Q


#====================================================
# Plot Ho-Lee Short-Rate Lattice
#====================================================
def plot_ho_lee_lattice(sigma, T, n_steps, P0):
    r_tree, _, _ = ho_lee_lattice(sigma=sigma, T=T, n_steps=n_steps, P0=P0)

    dt = T / n_steps
    fig, ax = plt.subplots(figsize=(12, 7))

    for n in range(n_steps + 1):
        time = n * dt

        for j in range(n + 1):
            r = r_tree[n, j]
            ax.scatter(time, r, s=20)

            if n < n_steps:
                time_next = (n + 1) * dt
                ax.plot([time, time_next], [r, r_tree[n + 1, j]], linewidth=0.7)
                ax.plot([time, time_next], [r, r_tree[n + 1, j + 1]], linewidth=0.7)

    ax.set_title("Ho-Lee Short-Rate Lattice")
    ax.set_xlabel("Time")
    ax.set_ylabel("Short Rate")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()