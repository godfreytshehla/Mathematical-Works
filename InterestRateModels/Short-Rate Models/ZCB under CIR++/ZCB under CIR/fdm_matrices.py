import numpy as np

#============================================
# Disretized Differential Operator Matrices
#============================================ 
def matrix_L(Nr, r_grid, a, b, sigma):
    dr = r_grid[1] - r_grid[0]
    def L_disc():
        return -np.diag(r_grid)
    def L_r():
        L = np.zeros((Nr, Nr))
        for i in range(1, Nr - 1):
            coeff = a * (b - r_grid[i])
            L[i, i - 1] = -coeff / (2 * dr)
            L[i, i + 1] =  coeff / (2 * dr)
        return L
    def L_rr():
        L = np.zeros((Nr, Nr))
        for i in range(1, Nr - 1):
            coeff = 0.5 * r_grid[i] * sigma**2 / dr**2
            L[i, i - 1] =  coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] =  coeff
        return L
    return L_disc() + L_r() + L_rr()