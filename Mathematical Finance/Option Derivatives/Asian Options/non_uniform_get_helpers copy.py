import numpy as np
from scipy.sparse import diags, eye, kron



#============================================
# Geometric Brownian Motion Paths (BSM)
#============================================
def gbm_paths(S0, T, r, q, sigma, n_time, n_sim, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_time
    Z = rng.standard_normal((n_time, n_sim))
    increments = (
        (r - q - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )
    S = np.empty((n_time + 1, n_sim))
    S[0] = S0
    S[1:] = S0 * np.exp(np.cumsum(increments, axis=0))
    return S


#============================================
# Non-Uniform S Grid
#============================================
def S_grid(Smax, X, Ns):

    d1 = X / 100

    xi_min = np.arcsinh(-X / d1)
    xi_max = np.arcsinh((Smax - X) / d1)

    xi = np.linspace(xi_min, xi_max, Ns)

    S = X + d1 * np.sinh(xi)

    return S


#============================================
# Non-Uniform I Grid
#============================================
def I_grid(Imax, X, T, Ni):

    I_star = X * T
    d1 = I_star / 100

    xi_min = np.arcsinh(-I_star / d1)
    xi_max = np.arcsinh((Imax - I_star) / d1)

    xi = np.linspace(xi_min, xi_max, Ni)

    I = I_star + d1 * np.sinh(xi)

    return I

#============================================
# Discretized Differential Operator (BSM)
#============================================
def bsm_matrix_L(r, q, sigma, S_grid):
    NS = len(S_grid)
    dS = S_grid[1] - S_grid[0]
    #----------------------------------------
    # Zeroth Derivative
    #----------------------------------------
    def zeroth_derivative():
        return np.eye(NS)
    #----------------------------------------
    # First Derivative
    #----------------------------------------
    def first_derivative():
        return (
            np.diag(-np.ones(NS - 1), -1)
            + np.diag(np.ones(NS - 1), 1)
        ) / (2 * dS)
    #----------------------------------------
    # Second Derivative
    #----------------------------------------
    def second_derivative():
        return (
            np.diag(np.ones(NS - 1), -1)
            - 2 * np.diag(np.ones(NS))
            + np.diag(np.ones(NS - 1), 1)
        ) / dS**2
    #----------------------------------------
    # BSM Operator
    #----------------------------------------
    L0 = -r * zeroth_derivative()
    L1 = (
        np.diag((r - q) * S_grid)
        @ first_derivative()
    )
    L2 = (
        np.diag(0.5 * sigma**2 * S_grid**2)
        @ second_derivative()
    )
    L = L0 + L1 + L2
    # Boundary rows excluded from PDE operator
    L[0, :] = 0
    L[-1, :] = 0
    return L

#============================================
# Arithmetic Asian Differential Operators
#============================================
def asian_arithmetic_matrix_A(S_grid, I_grid, r, q, sigma):

    Ns = len(S_grid)
    Ni = len(I_grid)

    dS = S_grid[1] - S_grid[0]
    dI = I_grid[1] - I_grid[0]

    #----------------------------------------
    # Derivative Matrices
    #----------------------------------------
    def first_derivative(N, dx):
        return diags(
            [-np.ones(N - 1), np.ones(N - 1)],
            [-1, 1],
            shape=(N, N)
        ) / (2 * dx)

    def second_derivative(N, dx):
        return diags(
            [np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)],
            [-1, 0, 1],
            shape=(N, N)
        ) / dx**2

    DS = first_derivative(Ns, dS)
    DSS = second_derivative(Ns, dS)
    DI = first_derivative(Ni, dI)

    IS = eye(Ns)
    II = eye(Ni)

    #----------------------------------------
    # S Direction
    #----------------------------------------
    LS = (
        diags((r - q) * S_grid) @ DS
        + diags(0.5 * sigma**2 * S_grid**2) @ DSS
        - r * IS
    )

    A1 = kron(LS, II, format="csr")

    #----------------------------------------
    # I Direction
    #----------------------------------------
    A2 = kron(
        diags(S_grid),
        DI,
        format="csr"
    )

    return A1, A2

#============================================
# Arithmetic Asian Differential Operators
# Non-Uniform Grid
#============================================
def asian_arithmetic_non_uniform_matrix_A(S_grid, I_grid, r, q, sigma):

    Ns = len(S_grid)
    Ni = len(I_grid)

    #----------------------------------------
    # First Derivative
    # Central Difference
    #----------------------------------------
    def first_derivative(grid):

        N = len(grid)

        lower = np.zeros(N - 1)
        main  = np.zeros(N)
        upper = np.zeros(N - 1)

        for i in range(1, N - 1):

            hm = grid[i] - grid[i - 1]
            hp = grid[i + 1] - grid[i]

            lower[i - 1] = -hp / (hm * (hm + hp))
            main[i]      = (hp - hm) / (hm * hp)
            upper[i]     = hm / (hp * (hm + hp))

        return diags([lower, main, upper], [-1, 0, 1], format="csr")

    #----------------------------------------
    # Second Derivative
    #----------------------------------------
    def second_derivative(grid):

        N = len(grid)

        lower = np.zeros(N - 1)
        main  = np.zeros(N)
        upper = np.zeros(N - 1)

        for i in range(1, N - 1):

            hm = grid[i] - grid[i - 1]
            hp = grid[i + 1] - grid[i]

            lower[i - 1] = 2 / (hm * (hm + hp))
            main[i]      = -2 / (hm * hp)
            upper[i]     = 2 / (hp * (hm + hp))

        return diags([lower, main, upper], [-1, 0, 1], format="csr")

    #----------------------------------------
    # Forward Derivative
    # I Direction
    #----------------------------------------
    def forward_derivative(grid):

        N = len(grid)

        main  = np.zeros(N)
        upper = np.zeros(N - 1)

        for i in range(N - 1):

            h = grid[i + 1] - grid[i]

            main[i]  = -1 / h
            upper[i] =  1 / h

        return diags([main, upper], [0, 1], format="csr")

    #----------------------------------------
    # Derivative Matrices
    #----------------------------------------
    DS  = first_derivative(S_grid)
    DSS = second_derivative(S_grid)
    DI  = forward_derivative(I_grid)

    IS = eye(Ns, format="csr")
    II = eye(Ni, format="csr")

    #----------------------------------------
    # S Direction
    #----------------------------------------
    LS = (
        diags((r - q) * S_grid) @ DS
        + diags(0.5 * sigma**2 * S_grid**2) @ DSS
        - r * IS
    )

    A1 = kron(LS, II, format="csr")

    #----------------------------------------
    # I Direction
    #----------------------------------------
    A2 = kron(
        diags(S_grid),
        DI,
        format="csr"
    )

    return A1, A2