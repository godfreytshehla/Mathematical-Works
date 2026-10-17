import numpy as np

#============================================
# Discretized Differential Operator (HW)
#============================================
def matrix_L_hw(Nr, r_grid, a, theta_t, sigma):
    dr = r_grid[1] - r_grid[0]
    # -----------------------------
    # discounting term: -r f
    # -----------------------------
    def L_disc():
        return -np.diag(r_grid)
    # -----------------------------
    # drift term: (theta - a r) f_r
    # central difference
    # -----------------------------
    def L_r():
        L = np.zeros((Nr, Nr))
        for i in range(1, Nr - 1):
            drift = theta_t - a * r_grid[i]
            L[i, i - 1] = -drift / (2 * dr)
            L[i, i + 1] =  drift / (2 * dr)
        return L
    # -----------------------------
    # diffusion term: (sigma^2 / 2) f_rr
    # constant coefficient
    # -----------------------------
    def L_rr():
        L = np.zeros((Nr, Nr))
        coeff = 0.5 * sigma**2 / dr**2
        for i in range(1, Nr - 1):
            L[i, i - 1] = coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] = coeff
        return L
    return L_disc() + L_r() + L_rr()