from asian_pricer import asian_arithmetic_mc_price, asian_arithmetic_non_uniform_do_price


# ============================================
# Model Parameters
# ============================================
S0 = 100.0
X = 100.0
T = 1.0
r = 0.05
q = 0.00
sigma = 0.20

option_type = "call"


# ============================================
# Monte Carlo Parameters
# ============================================
n_time = 252
n_sim = 100_000
seed = 42


# ============================================
# Douglas ADI Parameters
# ============================================
Nt = 100
Ns = 100
Ni = 100
theta = 0.5

Smax = 4.0 * max(S0, X)
Imax = Smax * T


# ============================================
# Monte Carlo
# ============================================
mc_price, mc_se, ci_lower, ci_upper = asian_arithmetic_mc_price(
    S0, X, T, r, q, sigma, n_time, n_sim, option_type, seed
)


# ============================================
# Non-Uniform Douglas ADI
# ============================================
do_price = asian_arithmetic_non_uniform_do_price(
    S0, X, T, r, q, sigma, Nt, Ns, Ni, Smax, Imax, option_type, theta
)


# ============================================
# Results
# ============================================
difference = abs(mc_price - do_price)

print(f"Arithmetic Asian {option_type.capitalize()}")
print("-" * 48)
print(f"Monte Carlo             : {mc_price:.6f}")
print(f"MC Standard Error       : {mc_se:.6f}")
print(f"MC 95% CI               : [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"Non-Uniform Douglas ADI : {do_price:.6f}")
print(f"Absolute Difference     : {difference:.6f}")