import numpy as np

from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.interpolate import RegularGridInterpolator

from get_helpers import gbm_paths, S_grid, I_grid, asian_arithmetic_non_uniform_matrix_A


# ============================================
# Arithmetic Asian Option - Monte Carlo
# ============================================
def asian_arithmetic_mc_price(S0, X, T, r, q, sigma, n_time, n_sim, option_type="call", seed=None):

    paths = gbm_paths(S0, T, r, q, sigma, n_time, n_sim, seed)

    # Arithmetic average over monitoring dates t1,...,tN
    average = np.mean(paths[1:], axis=0)

    if option_type.lower() == "call":
        payoff = np.maximum(average - X, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(X - average, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-r * T) * payoff

    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(n_sim)

    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return price, std_error, ci_lower, ci_upper


# ============================================
# Arithmetic Asian Option - Non-Uniform Douglas ADI
# ============================================
def asian_arithmetic_non_uniform_do_price(S0, X, T, r, q, sigma, Nt=100, Ns=100, Ni=100, Smax=None, Imax=None, option_type="call", theta=0.5):

    if Smax is None:
        Smax = 4.0 * max(S0, X)

    if Imax is None:
        Imax = Smax * T

    S = S_grid(Smax, X, Ns)
    I = I_grid(Imax, X, T, Ni)

    dt = T / Nt
    m = Ns * Ni

    A1, A2 = asian_arithmetic_non_uniform_matrix_A(S, I, r, q, sigma)
    A = A1 + A2

    Id = identity(m, format="csc")

    M1 = Id - theta * dt * A1
    M2 = Id - theta * dt * A2

    solve_M1 = factorized(M1.tocsc())
    solve_M2 = factorized(M2.tocsc())

    # Terminal payoff
    average = I / T

    if option_type.lower() == "call":
        payoff = np.maximum(average - X, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(X - average, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    U = np.tile(payoff, Ns)

    # Douglas ADI
    for _ in range(Nt):

        Y0 = U + dt * (A @ U)

        Y1_rhs = Y0 - theta * dt * (A1 @ U)
        Y1 = solve_M1(Y1_rhs)

        Y2_rhs = Y1 - theta * dt * (A2 @ U)
        U = solve_M2(Y2_rhs)

    U_grid = U.reshape(Ns, Ni, order="C")

    interpolator = RegularGridInterpolator((S, I), U_grid)

    return float(interpolator([[S0, 0.0]])[0])