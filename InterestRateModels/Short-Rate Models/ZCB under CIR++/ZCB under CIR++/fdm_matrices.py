import numpy as np

#============================================
# Disretized Differential Operator Matrices
#============================================ 
def matrix_L(Nx, x_grid, a, b, sigma):
    dx = x_grid[1] - x_grid[0]
    def L_disc():
        return -np.diag(x_grid)
    def L_x():
        L = np.zeros((Nx, Nx))
        for i in range(1, Nx - 1):
            coeff = a * (b - x_grid[i])
            L[i, i - 1] = -coeff / (2 * dx)
            L[i, i + 1] =  coeff / (2 * dx)
        return L
    def L_xx():
        L = np.zeros((Nx, Nx))
        for i in range(1, Nx - 1):
            coeff = 0.5 * x_grid[i] * sigma**2 / dx**2
            L[i, i - 1] =  coeff
            L[i, i]     = -2 * coeff
            L[i, i + 1] =  coeff
        return L
    return L_disc() + L_x() + L_xx()