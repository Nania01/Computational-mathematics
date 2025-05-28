import numpy as np
import pandas as pd
from tabulate import tabulate


def separate_roots(f: callable, a: float, b: float, bins: int) -> list:
    """
    Separates the interval [a, b] into subintervals where the function f(x) changes sign.

    This function detects root brackets by dividing the interval [a, b] into smaller
    segments and checking where the function f changes sign. According to the
    Bolzano–Cauchy theorem, a sign change in a continuous function implies at least
    one root in that interval.

    Args:
        f (callable): Function for which roots are being located. Must be continuous on [a, b].
        a (float): Left boundary of the interval.
        b (float): Right boundary of the interval.
        bins (int): Number of subintervals to divide [a, b] into.

    Returns:
        list: A list of tuples, where each tuple (x_i, x_{i+1}) defines an interval
              in which the function changes sign and a root is expected to exist.
    """
    x = np.linspace(a, b, bins)
    brackets = []
    for i in range(len(x) - 1):
        if np.sign(f(x[i])) != np.sign(f(x[i + 1])):
            brackets.append((x[i], x[i + 1]))
    return brackets


def newton_method(f: callable, df: callable, x0: float, eps: float) -> float:
    """
    Finds a root of the equation f(x) = 0 using the Newton-Raphson method (method of tangents).

    This method uses the first derivative of the function to iteratively approach a root.
    Convergence is fast if the initial approximation is close to the actual root and
    f(x) * f''(x) > 0 in the starting point.

    Args:
        f (callable): Function for which the root is being sought.
        df (callable): First derivative of the function f(x).
        x0 (float): Initial approximation.
        eps (float): Tolerance for stopping criterion (based on difference between iterations).

    Returns:
        float: Approximated root of the function with given accuracy.
    """
    x_prev = x0
    while True:
        x_next = x_prev - f(x_prev) / df(x_prev)
        if abs(x_next - x_prev) < eps:
            break
        x_prev = x_next
    return x_next


def secant_method(f: callable, x0: float, x1: float, eps: float) -> float:
    """
    Finds a root of the equation f(x) = 0 using the secant method (method of chords).

    This method does not require the derivative of the function. It approximates the root
    by building secants through two consecutive approximations.

    Args:
        f (callable): Function for which the root is being sought.
        x0 (float): First initial approximation.
        x1 (float): Second initial approximation.
        eps (float): Tolerance for stopping criterion (based on difference between iterations).

    Returns:
        float: Approximated root of the function with given accuracy.
    """
    while True:
        if f(x1) == f(x0):
            break
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < eps:
            break
        x0, x1 = x1, x2
    return x2


def choose_initial_points(f: callable, d2f: callable, a: float, b: float) -> tuple:
    """
    Chooses proper starting points for the Newton and secant methods based on second derivative.

    Selects the endpoint of the interval where the condition f(x) * f''(x) > 0 holds
    as the starting point for Newton's method to ensure convergence.

    Args:
        f (callable): Function for which the root is being sought.
        d2f (callable): Second derivative of the function f(x).
        a (float): Left endpoint of the interval.
        b (float): Right endpoint of the interval.

    Returns:
        tuple: (x_newton, x_secant), where:
            x_newton — start point for Newton's method,
            x_secant — other endpoint for the secant method.

    Raises:
        ValueError: If f(x) * f''(x) <= 0 at both endpoints.
    """
    fa, fb = f(a), f(b)
    d2fa, d2fb = d2f(a), d2f(b)
    if fa * d2fa > 0:
        return a, b
    elif fb * d2fb > 0:
        return b, a


f = lambda x: 0.5 * x**2 - np.cos(2 * x)
df = lambda x: x + 2 * np.sin(2 * x)
d2f = lambda x: 1 + 4 * np.cos(2 * x)

a, b = -5, 5
eps = 1e-6
intervals = separate_roots(f, a, b, bins=1000)

results = []

for a_i, b_i in intervals:
    x_newton, x_secant = choose_initial_points(f, d2f, a_i, b_i)
    root_newton = newton_method(f, df, x_newton, eps)
    root_secant = secant_method(f, x_secant, x_newton, eps)

    results.append((
        f"[{a_i:.6f}, {b_i:.6f}]",
        root_newton,
        root_secant
    ))

df = pd.DataFrame(results, columns=["Интервал", "Корень (метод касательных)", "Корень (метод хорд)"])
print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".10f", showindex=False))
