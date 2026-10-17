import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

#=====================================
# Grid Construction
#=====================================
def S_grid(Smax, X, Ns):
    d1 = X / 250
    xi_min = np.arcsinh(-X / d1)
    xi_max = np.arcsinh((Smax - X) / d1)
    xi = np.linspace(xi_min, xi_max, Ns)
    S = X + d1 * np.sinh(xi)
    return S

def v_grid(vmax, Nv):
    d2 = vmax / 250
    xi_max = np.arcsinh(vmax / d2)
    xi = np.linspace(0, xi_max, Nv)
    v = d2 * np.sinh(xi)
    return v

# #===========================================
# # SABR Mixed Differential Operator
# #===========================================
# def matrix_L0(F, nu, beta, rho, sigma):
#     NF = len(F)
#     Nnu = len(nu)
#     L0 = lil_matrix((NF * Nnu, NF * Nnu))
#     def idx(i, j):
#         return i * Nnu + j
#     for i in range(1, NF - 1):
#         dF_im1 = F[i] - F[i - 1]
#         dF_i = F[i + 1] - F[i]
#         dF_sum = dF_im1 + dF_i
#         for j in range(1, Nnu - 1):
#             dnu_jm1 = nu[j] - nu[j - 1]
#             dnu_j = nu[j + 1] - nu[j]
#             dnu_sum = dnu_jm1 + dnu_j
#             coeff = (
#                 rho * sigma * nu[j]**2 * F[i]**beta
#                 / (dF_sum * dnu_sum)
#             )
#             L0[idx(i, j), idx(i+1, j+1)] +=  coeff
#             L0[idx(i, j), idx(i+1, j-1)] += -coeff
#             L0[idx(i, j), idx(i-1, j+1)] += -coeff
#             L0[idx(i, j), idx(i-1, j-1)] +=  coeff
#     return L0.tocsr()


#===========================================
# SABR Mixed Differential Operator
# Non-uniform grid
#===========================================
def matrix_L0(F, nu, beta, rho, sigma):
    NF = len(F)
    Nnu = len(nu)
    L0 = lil_matrix((NF * Nnu, NF * Nnu))
    def idx(i, j):
        return i * Nnu + j
    for i in range(1, NF - 1):
        hm = F[i] - F[i - 1]
        hp = F[i + 1] - F[i]
        # First derivative weights in F
        a_m = -hp / (hm * (hm + hp))
        a_0 = (hp - hm) / (hm * hp)
        a_p = hm / (hp * (hm + hp))
        for j in range(1, Nnu - 1):
            km = nu[j] - nu[j - 1]
            kp = nu[j + 1] - nu[j]
            # First derivative weights in nu
            b_m = -kp / (km * (km + kp))
            b_0 = (kp - km) / (km * kp)
            b_p = km / (kp * (km + kp))
            coeff = (
                rho
                * sigma
                * nu[j]**2
                * F[i]**beta
            )
            # 9-point tensor-product stencil
            a = [a_m, a_0, a_p]
            b = [b_m, b_0, b_p]
            for di, ai in zip([-1, 0, 1], a):
                for dj, bj in zip([-1, 0, 1], b):
                    L0[idx(i, j), idx(i + di, j + dj)] += (
                        coeff * ai * bj
                    )
    return L0.tocsr()

#===========================================
# SABR Forward Differential Operator
#===========================================
def matrix_L1(F, nu, beta, r):
    NF = len(F)
    Nnu = len(nu)
    L1 = lil_matrix((NF * Nnu, NF * Nnu))
    def idx(i, j):
        return i * Nnu + j
    def L_FF():
        for i in range(1, NF - 1):
            dF_im1 = F[i] - F[i - 1]
            dF_i = F[i + 1] - F[i]
            dF_sum = dF_im1 + dF_i
            for j in range(Nnu):
                coeff = 0.5 * nu[j]**2 * F[i]**(2 * beta)
                gamma_minus = 2 * coeff / (dF_im1 * dF_sum)
                gamma_0 = -2 * coeff / (dF_im1 * dF_i)
                gamma_plus = 2 * coeff / (dF_i * dF_sum)
                L1[idx(i, j), idx(i-1, j)] += gamma_minus
                L1[idx(i, j), idx(i, j)] += gamma_0
                L1[idx(i, j), idx(i+1, j)] += gamma_plus
    def L_I():
        for i in range(NF):
            for j in range(Nnu):
                L1[idx(i, j), idx(i, j)] += -0.5 * r
    L_FF()
    L_I()
    return L1.tocsr()

#===========================================
# SABR Volatility Differential Operator
#===========================================
def matrix_L2(F, nu, sigma, r):
    NF = len(F)
    Nnu = len(nu)
    L2 = lil_matrix((NF * Nnu, NF * Nnu))
    def idx(i, j):
        return i * Nnu + j
    def L_nunu():
        for i in range(NF):
            for j in range(1, Nnu - 1):
                dnu_jm1 = nu[j] - nu[j - 1]
                dnu_j = nu[j + 1] - nu[j]
                dnu_sum = dnu_jm1 + dnu_j
                coeff = 0.5 * sigma**2 * nu[j]**2
                gamma_minus = 2 * coeff / (dnu_jm1 * dnu_sum)
                gamma_0 = -2 * coeff / (dnu_jm1 * dnu_j)
                gamma_plus = 2 * coeff / (dnu_j * dnu_sum)
                L2[idx(i, j), idx(i, j-1)] += gamma_minus
                L2[idx(i, j), idx(i, j)] += gamma_0
                L2[idx(i, j), idx(i, j+1)] += gamma_plus
    def L_I():
        for i in range(NF):
            for j in range(Nnu):
                L2[idx(i, j), idx(i, j)] += -0.5 * r
    L_nunu()
    L_I()
    return L2.tocsr()

#===========================================
# SABR Differential Operator
#===========================================
def matrix_L_sabr(F, nu, beta, rho, sigma, r):
    L0 = matrix_L0(F, nu, beta, rho, sigma)
    L1 = matrix_L1(F, nu, beta, r)
    L2 = matrix_L2(F, nu, sigma, r)
    return L0 + L1 + L2


#=====================================================
# SABR model path generation using discretization
#=====================================================
def sabr_paths(
    F0, nu0, beta, sigma, rho,
    T, n_time, n_sim, seed=None
):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_time
    F = np.zeros((n_sim, n_time + 1))
    nu = np.zeros((n_sim, n_time + 1))
    F[:, 0] = F0
    nu[:, 0] = nu0
    for t in range(1, n_time + 1):
        z1 = np.random.normal(size=n_sim)
        z2 = np.random.normal(size=n_sim)
        z_F = z1
        z_nu = (
            rho * z1
            + np.sqrt(1 - rho**2) * z2
        )
        F[:, t] = (
            F[:, t - 1]
            + nu[:, t - 1]
            * np.maximum(F[:, t - 1], 0.0)**beta
            * np.sqrt(dt)
            * z_F
        )
        nu[:, t] = (
            nu[:, t - 1]
            * np.exp(
                -0.5 * sigma**2 * dt
                + sigma * np.sqrt(dt) * z_nu
            )
        )
    return F