import numpy as np
from scipy.optimize import minimize


class vasicek_calibrator:
    def __init__(self,
                 T_mkt,
                 price_mkt):
        self.T_mkt = np.asarray(T_mkt)
        self.price_mkt = np.asarray(price_mkt)

        self.params = None
        self.result = None
    # =========================
    # Vasicek ZCB price
    # =========================
    @staticmethod
    def vasicek_zcb_price(params, T):
        a, b, sigma, r0 = params
        T = np.asarray(T)
        # B(T)
        Bt = (1 - np.exp(-a * T)) / a
        # A(T)
        term1 = (
            b - sigma**2 / (2 * a**2)
        ) * (Bt - T)
        term2 = (
            sigma**2 / (4 * a)
        ) * Bt**2
        At = term1 - term2
        # ZCB price
        return np.exp(At - Bt * r0)

    # =========================
    # Objective function
    # =========================
    def loss(self, params):
        a, b, sigma, r0 = params
        if a <= 0 or sigma <= 0:
            return 1e10
        price_model = self.vasicek_zcb_price(
            params,
            self.T_mkt
        )
        rel_err = (
            (price_model - self.price_mkt)
            / self.price_mkt
        )
        return np.sum(rel_err**2)

    # =========================
    # Calibration
    # =========================

    def calibrate(self,
                  x0=None,
                  bounds=None):
        if x0 is None:
            x0 = np.array([
                0.5,
                0.04,
                0.01,
                0.03
            ])
        if bounds is None:
            bounds = [
                (1e-4, 5.0),   # a
                (0.0, 0.20),   # b
                (1e-4, 0.20),  # sigma
                (0.0, 0.20)    # r0
            ]
        self.result = minimize(
            self.loss,
            x0,
            bounds=bounds,
            method="L-BFGS-B"
        )
        self.params = self.result.x
        return self.params

    # =========================
    # Pretty print
    # =========================
    def summary(self):
        if self.params is None:
            print("Model not calibrated.")
            return
        param_names = ["a", "b", "sigma", "r0"]
        print("\nCalibrated Vasicek Parameters")
        print("-" * 35)
        print(f"{'Parameter':<10} {'Value':>15}")
        print("-" * 35)
        for name, value in zip(param_names, self.params):
            print(f"{name:<10} {value:>15.6f}")
        print("-" * 35)
        print(f"{'Objective':<10} {self.result.fun:>15.6e}")