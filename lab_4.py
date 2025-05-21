import numpy as np
from scipy.integrate import quad
import pandas as pd
from tabulate import tabulate


def left_rectangles(f: np.ndarray, a: float, b: float, n: int) -> float:
    """
    Computes the integral using the left rectangle (left Riemann sum) method.

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of subintervals.

    Returns:
        float: Approximated integral value.
    """
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))


def right_rectangles(f: np.ndarray, a: float, b: float, n: int) -> float:
    """
    Computes the integral using the right rectangle method.

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of subintervals.

    Returns:
        float: Approximated integral value.
    """
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))


def middle_rectangles(f: np.ndarray, a: float, b: float, n: int) -> float:
    """
    Computes the integral using the midpoint rectangle method.

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of subintervals.

    Returns:
        float: Approximated integral value.
    """
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))


def trapezoidal(f: np.ndarray, a: float, b: float, n: int) -> float:
    """
    Computes the integral using the trapezoidal rule.

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of subintervals.

    Returns:
        float: Approximated integral value.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])


def simpson(f: np.ndarray, a: float, b: float, n: int) -> float:
    """
    Computes the integral using Simpson's rule (parabolic approximation).

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of subintervals (must be even).

    Returns:
        float: Approximated integral value.

    Raises:
        ValueError: If n is not even.
    """
    if n % 2 != 0:
        raise ValueError("Для формулы Симпсона n должно быть чётным")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])


def weddle_n6(f: np.ndarray, a: float, b: float) -> float:
    """
    Computes the integral using Weddle's rule (specific case for n=6).

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.

    Returns:
        float: Approximated integral value.
    """
    n = 6
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    coeffs = np.array([1, 5, 1, 6, 1, 5, 1])
    return 3 * h / 10 * np.dot(coeffs, y)


def newton_cotes_n5(f: np.ndarray, a: float, b: float) -> float:
    """
    Computes the integral using Newton-Cotes formula of degree 5 (closed form).

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.

    Returns:
        float: Approximated integral value.
    """
    n = 5
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    c = np.array([19, 75, 50, 50, 75, 19]) / 288
    return (b - a) * np.dot(c, y)


def gauss_n4(f: np.ndarray, a: float, b: float) -> float:
    """
    Computes the integral using Gauss quadrature with n=4 nodes.

    Args:
        f (np.ndarray): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.

    Returns:
        float: Approximated integral value.
    """
    t = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
    c = np.array([0.347855, 0.652145, 0.652145, 0.347855])
    x = (b + a) / 2 + (b - a) / 2 * t
    y = f(x)
    return (b - a) / 2 * np.dot(c, y)


n = 10
a, b = 0.1, 0.6
f = lambda x: x**2 - 0.5 * np.exp(-x)

exact_value, _ = quad(f, a, b)
print(f"\nТочное значение интеграла: {exact_value:.10f}")

results = [
    ("Левые прямоугольники", 10, left_rectangles(f, a, b, 10)),
    ("Правые прямоугольники", 10, right_rectangles(f, a, b, 10)),
    ("Средние прямоугольники", 10, middle_rectangles(f, a, b, 10)),
    ("Трапеции", 10, trapezoidal(f, a, b, 10)),
    ("Симпсон", 10, simpson(f, a, b, 10)),
    ("Веддля", 6, weddle_n6(f, a, b)),
    ("Ньютон-Котес", 5, newton_cotes_n5(f, a, b)),
    ("Гаусс", 4, gauss_n4(f, a, b)),
]

df = pd.DataFrame(results, columns=["Формула", "n", "Результат"])
df["Погрешность"] = np.abs(df["Результат"] - exact_value)

print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".10f", showindex=False))
