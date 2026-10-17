import numpy as np

def trapezoidal_rule(f, a, b, n):
    """
    Approximates the integral of f(x) from a to b using the trapezoidal rule.

    Parameters:
        f (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        n (int): The number of subintervals.

    Returns:
        float: The approximate value of the integral.
    """
    dx = (b - a) / n  # Width of each subinterval
    integral = 0.5 * (f(a) + f(b))  # Add the endpoints
    for i in range(1, n):
        integral += f(a + i * dx)
    integral *= dx  # Multiply by the width
    return integral

def simpsons_rule(f, a, b, n):
    """
    Approximates the integral of f(x) from a to b using Simpson's rule.

    Parameters:
        f (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        n (int): The number of subintervals (must be even).

    Returns:
        float: The approximate value of the integral.
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's rule.")
    dx = (b - a) / n  # Width of each subinterval
    integral = f(a) + f(b)  # Add the endpoints
    for i in range(1, n):
        x = a + i * dx
        if i % 2 == 0:
            integral += 2 * f(x)  # Even-indexed points
        else:
            integral += 4 * f(x)  # Odd-indexed points
    integral *= dx / 3  # Multiply by the width and divide by 3
    return integral