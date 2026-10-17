import numpy as np
from scipy.sparse import diags, eye, kron


# ============================================
# Geometric Brownian Motion Paths
# ============================================
def gbm_paths(S0, T, r, q, sigma, n_time, n_sim, seed=None):

    rng = np.random.default_rng(seed)
    dt = T / n_time
    Z = rng.standard_normal((n_time, n_sim))

    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    S = np.empty((n_time + 1, n_sim))
    S[0] = S0
    S[1:] = S0 * np.exp(np.cumsum(increments, axis=0))

    return S


# ============================================
# Non-Uniform S Grid
# ============================================
def S_grid(Smax, X, Ns):

    if Smax <= X:
        raise ValueError("Smax must be greater than X.")

    if Ns < 3:
        raise ValueError("Ns must be at least 3.")

    scale = X / 100.0

    xi_min = np.arcsinh(-X / scale)
    xi_max = np.arcsinh((Smax - X) / scale)

    xi = np.linspace(xi_min, xi_max, Ns)
    S = X + scale * np.sinh(xi)

    S[0] = 0.0
    S[-1] = Smax

    return S


# ============================================
# Non-Uniform I Grid
# ============================================
def I_grid(Imax, X, T, Ni):

    if Imax <= X * T:
        raise ValueError("Imax must be greater than X*T.")

    if Ni < 3:
        raise ValueError("Ni must be at least 3.")

    I_star = X * T
    scale = I_star / 100.0

    xi_min = np.arcsinh(-I_star / scale)
    xi_max = np.arcsinh((Imax - I_star) / scale)

    xi = np.linspace(xi_min, xi_max, Ni)
    I = I_star + scale * np.sinh(xi)

    I[0] = 0.0
    I[-1] = Imax

    return I


# ============================================
# Non-Uniform First Derivative
# ============================================
def first_derivative(grid):

    N = len(grid)

    lower = np.zeros(N - 1)
    main = np.zeros(N)
    upper = np.zeros(N - 1)

    for i in range(1, N - 1):
        hm = grid[i] - grid[i - 1]
        hp = grid[i + 1] - grid[i]

        lower[i - 1] = -hp / (hm * (hm + hp))
        main[i] = (hp - hm) / (hm * hp)
        upper[i] = hm / (hp * (hm + hp))

    return diags([lower, main, upper], [-1, 0, 1], shape=(N, N), format="csr")


# ============================================
# Non-Uniform Second Derivative
# ============================================
def second_derivative(grid):

    N = len(grid)

    lower = np.zeros(N - 1)
    main = np.zeros(N)
    upper = np.zeros(N - 1)

    for i in range(1, N - 1):
        hm = grid[i] - grid[i - 1]
        hp = grid[i + 1] - grid[i]

        lower[i - 1] = 2.0 / (hm * (hm + hp))
        main[i] = -2.0 / (hm * hp)
        upper[i] = 2.0 / (hp * (hm + hp))

    return diags([lower, main, upper], [-1, 0, 1], shape=(N, N), format="csr")


# ============================================
# Non-Uniform Forward Derivative
# ============================================
def forward_derivative(grid):

    N = len(grid)

    main = np.zeros(N)
    upper = np.zeros(N - 1)

    for i in range(N - 1):
        h = grid[i + 1] - grid[i]

        main[i] = -1.0 / h
        upper[i] = 1.0 / h

    return diags([main, upper], [0, 1], shape=(N, N), format="csr")


# ============================================
# Arithmetic Asian PDE Operators
# ============================================
def asian_arithmetic_non_uniform_matrix_A(S, I, r, q, sigma):

    Ns = len(S)
    Ni = len(I)

    DS = first_derivative(S)
    DSS = second_derivative(S)
    DI = forward_derivative(I)

    IS = eye(Ns, format="csr")
    II = eye(Ni, format="csr")

    LS = diags((r - q) * S) @ DS + diags(0.5 * sigma**2 * S**2) @ DSS - r * IS

    A1 = kron(LS, II, format="csr")
    A2 = kron(diags(S), DI, format="csr")

    return A1, A2