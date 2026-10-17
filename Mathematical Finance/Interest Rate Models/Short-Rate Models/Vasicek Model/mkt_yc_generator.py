import numpy as np

#===========================
# Yield curve generator
#===========================
def yield_curve(T,
                beta0=0.09,
                beta1=-0.06,
                beta2=0.015,
                beta3=-0.01,
                l1=1.2,
                l2=0.3):
    T = np.asarray(T, dtype=float)
    eps = 1e-12
    def f(T, lam):
        x = lam * T
        return (1 - np.exp(-x)) / (x + eps)
    # Nelson-Siegel components
    level = beta0
    slope = beta1 * f(T, l1)
    curvature1 = beta2 * (f(T, l1) - np.exp(-l1 * T))
    curvature2 = beta3 * (f(T, l2) - np.exp(-l2 * T))
    y = level + slope + curvature1 + curvature2
    return y