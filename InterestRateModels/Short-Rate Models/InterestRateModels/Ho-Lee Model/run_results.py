from functools import partial

from my_yield_curve import discount_curve, forward_rate
from helpers import (
    ho_lee_paths,
    ho_lee_theta,
    plot_ho_lee_lattice,
    market_zcb_price,
)
from zcb_pricer import (
    zcb_ho_lee_analytical,
    zcb_ho_lee_mc,
    zcb_ho_lee_fdm,
    zcb_ho_lee_lattice,
)
from zcb_option_pricer import (
    zcb_option_ho_lee_closed_form,
    zcb_option_ho_lee_mc,
    zcb_option_ho_lee_fdm,
    zcb_option_ho_lee_lattice,
)


#====================================================
# Ho-Lee Pricing Results
#====================================================
if __name__ == "__main__":

    # -----------------------------
    # Model parameters
    # -----------------------------
    sigma = 0.01
    t = 0.0
    T = 5.0
    N = 1.0

    # -----------------------------
    # ZCB option parameters
    # -----------------------------
    T_1 = 2.0
    T_2 = 5.0
    option_type = "call"

    # -----------------------------
    # Monte Carlo parameters
    # -----------------------------
    n_time = 500
    n_sim = 100_000
    seed = 42

    # -----------------------------
    # Finite difference parameters
    # -----------------------------
    Nr = 301
    Nt = 200
    r_min = -0.10
    r_max = 0.20

    # -----------------------------
    # Lattice parameters
    # -----------------------------
    n_steps = 500
    n_plot_steps = 10

    # -----------------------------
    # Initial short rate
    # -----------------------------
    r0 = forward_rate(0.0)

    # -----------------------------
    # Ho-Lee calibrated drift
    # -----------------------------
    theta = partial(
        ho_lee_theta,
        sigma=sigma,
        forward_rate=forward_rate,
    )

    #================================================
    # ZCB PRICING
    #================================================

    # -----------------------------
    # Market ZCB
    # -----------------------------
    market_price = market_zcb_price(
        T=T,
        P0=discount_curve,
        N=N,
    )

    # -----------------------------
    # Analytical ZCB
    # -----------------------------
    analytical_price = zcb_ho_lee_analytical(
        r_t=r0,
        sigma=sigma,
        t=t,
        T=T,
        P0=discount_curve,
        f0_t=forward_rate(t),
        N=N,
    )

    # -----------------------------
    # Monte Carlo ZCB
    # -----------------------------
    r_paths_zcb = ho_lee_paths(
        r0=r0,
        theta=theta,
        sigma=sigma,
        T=T - t,
        n_time=n_time,
        n_sim=n_sim,
        seed=seed,
    )

    mc_price = zcb_ho_lee_mc(
        r_paths=r_paths_zcb,
        t=t,
        T=T,
        N=N,
    )

    # -----------------------------
    # Implicit FDM ZCB
    # -----------------------------
    fdm_price = zcb_ho_lee_fdm(
        r0=r0,
        theta=theta,
        sigma=sigma,
        T=T,
        Nr=Nr,
        Nt=Nt,
        r_min=r_min,
        r_max=r_max,
        N=N,
    )

    # -----------------------------
    # Lattice ZCB
    # -----------------------------
    lattice_price = zcb_ho_lee_lattice(
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        P0=discount_curve,
        N=N,
    )

    # -----------------------------
    # ZCB pricing errors
    # -----------------------------
    analytical_error = analytical_price - market_price
    mc_error = mc_price - market_price
    fdm_error = fdm_price - market_price
    lattice_error = lattice_price - market_price

    #================================================
    # ZCB OPTION PRICING
    #================================================

    # -----------------------------
    # ATM-forward strike
    # -----------------------------
    X = discount_curve(T_2) / discount_curve(T_1)

    # -----------------------------
    # Closed-form ZCB option
    # -----------------------------
    option_closed_form = zcb_option_ho_lee_closed_form(
        sigma=sigma,
        t=t,
        T_1=T_1,
        T_2=T_2,
        K=X,
        P0=discount_curve,
        option_type=option_type,
        N=N,
    )

    # -----------------------------
    # Monte Carlo paths to T_1
    # -----------------------------
    r_paths_option = ho_lee_paths(
        r0=r0,
        theta=theta,
        sigma=sigma,
        T=T_1 - t,
        n_time=n_time,
        n_sim=n_sim,
        seed=seed,
    )

    # -----------------------------
    # Monte Carlo ZCB option
    # -----------------------------
    option_mc = zcb_option_ho_lee_mc(
        r_paths=r_paths_option,
        sigma=sigma,
        t=t,
        T_1=T_1,
        T_2=T_2,
        K=X,
        P0=discount_curve,
        f0_T1=forward_rate(T_1),
        option_type=option_type,
        N=N,
    )

    # -----------------------------
    # Implicit FDM ZCB option
    # -----------------------------
    option_fdm = zcb_option_ho_lee_fdm(
        r0=r0,
        theta=theta,
        sigma=sigma,
        T_1=T_1,
        T_2=T_2,
        K=X,
        P0=discount_curve,
        f0_T1=forward_rate(T_1),
        option_type=option_type,
        Nr=Nr,
        Nt=Nt,
        r_min=r_min,
        r_max=r_max,
        N=N,
    )

    # -----------------------------
    # Lattice ZCB option
    # -----------------------------
    option_lattice = zcb_option_ho_lee_lattice(
        sigma=sigma,
        T_1=T_1,
        T_2=T_2,
        K=X,
        P0=discount_curve,
        option_type=option_type,
        n_steps=n_steps,
        N=N,
    )

    # -----------------------------
    # ZCB option pricing errors
    # -----------------------------
    option_mc_error = option_mc - option_closed_form
    option_fdm_error = option_fdm - option_closed_form
    option_lattice_error = option_lattice - option_closed_form

    #================================================
    # ZCB RESULTS
    #================================================

    print("\n================================================")
    print("              HO-LEE ZCB PRICING")
    print("================================================")
    print(f"Maturity             : {T:.2f} years")
    print(f"Initial short rate    : {r0:.6f}")
    print(f"Volatility            : {sigma:.6f}")
    print("------------------------------------------------")
    print(f"Market ZCB            : {market_price:.8f}")
    print(f"Analytical ZCB        : {analytical_price:.8f}")
    print(f"Monte Carlo ZCB       : {mc_price:.8f}")
    print(f"Implicit FDM ZCB      : {fdm_price:.8f}")
    print(f"Lattice ZCB           : {lattice_price:.8f}")
    print("------------------------------------------------")
    print(f"Analytical error      : {analytical_error:.8e}")
    print(f"Monte Carlo error     : {mc_error:.8e}")
    print(f"Implicit FDM error    : {fdm_error:.8e}")
    print(f"Lattice error         : {lattice_error:.8e}")
    print("================================================")

    #================================================
    # ZCB OPTION RESULTS
    #================================================

    print("\n================================================")
    print("          HO-LEE EUROPEAN ZCB OPTION")
    print("================================================")
    print(f"Option type           : {option_type.upper()}")
    print(f"Option expiry T_1     : {T_1:.2f} years")
    print(f"Bond maturity T_2     : {T_2:.2f} years")
    print(f"Strike X              : {X:.8f}")
    print(f"Volatility            : {sigma:.6f}")
    print("------------------------------------------------")
    print(f"Closed-form price     : {option_closed_form:.8f}")
    print(f"Monte Carlo price     : {option_mc:.8f}")
    print(f"Implicit FDM price    : {option_fdm:.8f}")
    print(f"Lattice price         : {option_lattice:.8f}")
    print("------------------------------------------------")
    print(f"Monte Carlo error     : {option_mc_error:.8e}")
    print(f"Implicit FDM error    : {option_fdm_error:.8e}")
    print(f"Lattice error         : {option_lattice_error:.8e}")
    print("================================================\n")

    #================================================
    # Plot Ho-Lee Lattice
    #================================================

    plot_ho_lee_lattice(
        sigma=sigma,
        T=T_2,
        n_steps=n_plot_steps,
        P0=discount_curve,
    )