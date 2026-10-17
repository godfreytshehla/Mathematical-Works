from cir_paths_generator import*
from fdm_matrices import*
from scipy.linalg import solve


############################################################# Crank-Nicolson Schemes ###########################################################

#====================================================
# Zero-Coupon Bond Valuation: Crank-Nicolson Scheme
#====================================================
def callable_zcb_cir_cranknicolson_price(r0, a, b, sigma, T, tc, Nt=150, Nr=200, N=1.0, Kc=0.9):
    if not (0 < tc < T):
        raise ValueError("Call date tc must satisfy 0 < tc < T.")
    rmax = b + 5 * sigma + 3 * r0 # maybe this should be an input?
    r_grid = np.linspace(0, rmax, Nr)
    dr = r_grid[1] - r_grid[0]
    dt = T / Nt
    L = matrix_L(Nr, r_grid, a, b, sigma)
    I = np.eye(Nr)
    M1, M2 = I - 0.5 * dt * L, I + 0.5 * dt * L
    V = np.full(Nr, N)
    n_call = int(tc / dt)
    for n in reversed(range(Nt)):
        t = n * dt
        V = solve(M1, M2 @ V)
        V[0] = N * np.exp(-r_grid[0] * (T - t))   # r -> 0
        V[-1] = 0                                 # r -> r_max
        # enforce call at tc
        if n == n_call:
            V = np.minimum(V, Kc) # embedded optionality
        #V = np.minimum(V, Kc) # embedded optionality
    return np.interp(r0, r_grid, V)

def callable_zcb_cir_cn(
    r0, a, b, sigma, T, t_call_dates,
    Nt=150, Nr=200, N=1.0, Kc=0.9,
    exercise_type="European"
):
    """
    Price a callable zero-coupon bond under CIR using Crank-Nicolson.
    
    Parameters
    ----------
    r0 : float
        Initial short rate
    a, b, sigma : floats
        CIR model parameters
    T : float
        Bond maturity
    t_call_dates : float or list of floats
        Exercise date(s). Single float for European, list for Bermudan.
    Nt, Nr : int
        Time and rate steps
    N : float
        Face value of the bond
    Kc : float
        Call price
    exercise_type : str
        "European" or "Bermudan"
    """
    dt = T / Nt
    if exercise_type == "European":
        if not (0 < t_call_dates < T):
            raise ValueError("European call date must satisfy 0 < t_call_dates < T.")
        call_indices = [int(t_call_dates / dt)]
    elif exercise_type == "Bermudan":
        if any((tc <= 0 or tc >= T) for tc in t_call_dates):
            raise ValueError("All Bermudan call dates must satisfy 0 < t_call_dates < T.")
        call_indices = [int(tc / dt) for tc in t_call_dates]
    else:
        raise ValueError('exercise_type must be "European" or "Bermudan"')
    rmax = b + 5 * sigma + 3 * r0
    r_grid = np.linspace(0, rmax, Nr)
    dr = r_grid[1] - r_grid[0]
    L = matrix_L(Nr, r_grid, a, b, sigma)  
    I = np.eye(Nr)
    M1, M2 = I - 0.5 * dt * L, I + 0.5 * dt * L
    V = np.full(Nr, N)
    for n in reversed(range(Nt)):
        t = n * dt
        V = solve(M1, M2 @ V)
        V[0] = N * np.exp(-r_grid[0] * (T - t))  # r -> 0 boundary
        V[-1] = 0                                # r -> rmax boundary
        if n in call_indices:
            V = np.minimum(V, Kc)               # apply call option
    return np.interp(r0, r_grid, V)

def embedded_call_option(r0, a, b, sigma, T, tc, Nt=150, Nr=200, N=1.0, Kc=0.9):
    if not (0 < tc < T):
        raise ValueError("Call date tc must satisfy 0 < tc < T.")
    rmax = b + 5 * sigma + 3 * r0 # maybe this should be an input?
    r_grid = np.linspace(0, rmax, Nr)
    dr = r_grid[1] - r_grid[0]
    dt = T / Nt
    L = matrix_L(Nr, r_grid, a, b, sigma)
    I = np.eye(Nr)
    M1, M2 = I - 0.5 * dt * L, I + 0.5 * dt * L
    V = np.full(Nr, N)
    n_call = int(tc / dt)
    for n in reversed(range(Nt)):
        t = n * dt
        V = solve(M1, M2 @ V)
        V[0] = N * np.exp(-r_grid[0] * (T - t))   # r -> 0
        V[-1] = 0                                 # r -> r_max
        # enforce call at tc
        if n == n_call:
            V = np.maximum(V - Kc, 0) # embedded European optionality
    return np.interp(r0, r_grid, V)


############################################################# Monte Carlo Simulation ############################################################

def phi(r):
    return np.vstack([np.ones_like(r), r, r**2, r**3, r**4]).T

def callable_zcb_cir_lsm(
    r0, a, b, sigma, T, t_call_dates,
    n_time=500, n_sim=10000, N=1.0, Kc=0.9,
    exercise_type="European"
):
    """
    Price a callable zero-coupon bond using LSM (Longstaff-Schwartz) under CIR.
    
    Parameters
    ----------
    r0 : float
        Initial short rate
    a, b, sigma : floats
        CIR parameters
    T : float
        Bond maturity
    t_call_dates : float or list of floats
        Exercise date(s). Single float for European, list for Bermudan.
    n_time, n_sim : int
        Number of time steps and simulations
    N : float
        Face value
    Kc : float
        Call price
    exercise_type : str
        "European" or "Bermudan"
    """
    dt = T / n_time
    # Convert call dates to time indices
    if exercise_type == "European":
        if not (0 < t_call_dates < T):
            raise ValueError("European call date must satisfy 0 < t_call_dates < T.")
        idx_call_dates = [int(t_call_dates / dt)]
    elif exercise_type == "Bermudan":
        if any((tc <= 0 or tc >= T) for tc in t_call_dates):
            raise ValueError("All Bermudan call dates must satisfy 0 < t_call_dates < T.")
        idx_call_dates = [int(tc / dt) for tc in t_call_dates]
    else:
        raise ValueError('exercise_type must be "European" or "Bermudan"')
    r_paths = cir_paths(r0, a, b, sigma, T, n_time, n_sim) 
    V = np.full(n_sim, N)
    for t in reversed(range(n_time)):
        r_t = r_paths[:, t]
        Y = V * np.exp(-r_t * dt)  # discounted continuation value
        A = phi(r_t)                # regression basis 
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation = A @ coeffs
        if t in idx_call_dates:
            V = np.where(Kc < continuation, Kc, Y)
        else:
            V = continuation
    return np.mean(V)