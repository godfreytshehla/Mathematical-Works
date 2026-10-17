import numpy as np

#====================================================
# Zero-Coupon Bond Valuation: Hull–White Analytical
#====================================================
def zcb_hw_analytical(r_t, a, sigma, t, T, P0, f0_t, N=1.0):
    """
    r_t   : short rate at time t
    a     : mean reversion
    sigma : volatility
    t     : valuation time
    T     : maturity
    P0    : function P0(T) = initial discount curve
    f0_t  : instantaneous forward rate f(0,t)
    N     : notional
    """
    # ---------------------------------------
    # B(t,T)
    # ---------------------------------------
    B = (1.0 - np.exp(-a * (T - t))) / a
    # ---------------------------------------
    # Initial curve inputs
    # ---------------------------------------
    P0_T = P0(T)
    P0_t = P0(t)
    # ---------------------------------------
    # Convexity adjustment term
    # ---------------------------------------
    sigma_term = (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * (B**2)
    # ---------------------------------------
    # A(t,T)
    # ---------------------------------------
    log_A = np.log(P0_T / P0_t) + B * f0_t - sigma_term
    A = np.exp(log_A)
    # ---------------------------------------
    # Bond price
    # ---------------------------------------
    return N * A * np.exp(-B * r_t)