import numpy as np
import pandas as pd
from math import sin, cos, exp
from tabulate import tabulate


def find_first_diff(nodes: list[float], x: float, f: callable) -> float:
    """
    Computes the first derivative of a function using the 3-point Lagrange interpolation.

    This function uses the central point and its immediate neighbors to estimate the
    first derivative at a given point based on the Lagrange polynomial differentiation.

    Args:
        nodes (list[float]): List of three nodes [x_{i-1}, x_i, x_{i+1}].
        x (float): The point at which to approximate the derivative.
        f (callable): The function whose derivative is being approximated.

    Returns:
        float: Approximated first derivative at the point x.
    """
    x_im1, x_i, x_ip1 = nodes

    f_im1 = f(x_im1)
    f_i = f(x_i)
    f_ip1 = f(x_ip1)

    l_im1_der = ((x - x_ip1) + (x - x_i)) / ((x_im1 - x_i) * (x_im1 - x_ip1))
    l_i_der = ((x - x_ip1) + (x - x_im1)) / ((x_i - x_im1) * (x_i - x_ip1))
    l_ip1_der = ((x - x_i) + (x - x_im1)) / ((x_ip1 - x_im1) * (x_ip1 - x_i))

    return f_im1 * l_im1_der + f_i * l_i_der + f_ip1 * l_ip1_der


def find_second_diff(nodes: list[float], f: callable) -> float:
    """
    Computes the second derivative of a function using the 3-point Lagrange interpolation.

    This method approximates the second derivative at the central node using the
    second-order formula derived from differentiating the Lagrange polynomial twice.

    Args:
        nodes (list[float]): List of three nodes [x_{i-1}, x_i, x_{i+1}].
        f (callable): The function whose second derivative is being approximated.

    Returns:
        float: Approximated second derivative at the central point.
    """
    x_im1, x_i, x_ip1 = nodes

    f_im1 = f(x_im1)
    f_i = f(x_i)
    f_ip1 = f(x_ip1)

    l_im1 = 1 / ((x_im1 - x_i) * (x_im1 - x_ip1))
    l_i = 1 / ((x_i - x_im1) * (x_i - x_ip1))
    l_ip1 = 1 / ((x_ip1 - x_im1) * (x_ip1 - x_i))

    return 2 * (f_im1 * l_im1 + f_i * l_i + f_ip1 * l_ip1)


f_1 = lambda x: x**2 + sin(x)
f_2 = lambda x: x**2 + 0.3 * exp(-x)
f_3 = lambda x: x**2 + cos(x)

f_1_first_diff = lambda x: 2*x + cos(x)
f_2_first_diff = lambda x: 2*x - 0.3 * exp(-x)
f_3_first_diff = lambda x: 2*x - sin(x)

f_1_second_diff = lambda x: 2 - sin(x)
f_2_second_diff = lambda x: 2 + 0.3 * exp(-x)
f_3_second_diff = lambda x: 2 - cos(x)

x_point = 0.55
nodes = [0.5, 0.55, 0.6]

results = [
    ("x^2 + sin(x)", 1, find_first_diff(nodes, x_point, f_1), f_1_first_diff(x_point)),
    ("x^2 + 0.3 * exp(-x)", 1, find_first_diff(nodes, x_point, f_2), f_2_first_diff(x_point)),
    ("x^2 + cos(x)", 1, find_first_diff(nodes, x_point, f_3), f_3_first_diff(x_point)),
    ("x^2 + sin(x)", 2, find_second_diff(nodes, f_1), f_1_second_diff(x_point)),
    ("x^2 + 0.3 * exp(-x)", 2, find_second_diff(nodes, f_2), f_2_second_diff(x_point)),
    ("x^2 + cos(x)", 2, find_second_diff(nodes, f_3), f_3_second_diff(x_point)),
]

df = pd.DataFrame(results, columns=["Функция", "Порядок производной", "Полином Лагранжа", "Истинное значение"])
df["Погрешность"] = np.abs(df["Полином Лагранжа"] - df["Истинное значение"])

print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".10f", showindex=False))
