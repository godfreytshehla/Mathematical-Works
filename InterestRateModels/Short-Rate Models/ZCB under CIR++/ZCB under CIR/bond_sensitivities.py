from zcb_pricer import zcb_cir_analytical
from callable_zcb_pricer import*

#==================================================
# Computes bond effective duration and convexity
#==================================================
def bond_risk_measures(P_func, r0, *args, eps=1e-4, **kwargs):
    """
    Compute effective duration and convexity for a bond.

    Parameters:
    - P_func: function returning bond price given r0
    - r0: current short rate
    - *args, **kwargs: extra parameters for P_func
    - eps: small perturbation in r0 for finite differences

    Returns:
    - duration: effective duration
    - convexity: effective convexity
    """
    P0 = P_func(r0, *args, **kwargs)
    P_plus = P_func(r0 + eps, *args, **kwargs)
    P_minus = P_func(r0 - eps, *args, **kwargs)
    # Central difference for first derivative -> duration
    duration = -(P_plus - P_minus) / (2 * eps * P0)
    # Central difference for second derivative -> convexity
    convexity = (P_plus - 2 * P0 + P_minus) / (eps**2 * P0)
    return duration, convexity