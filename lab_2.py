import numpy as np
import pandas as pd
import sympy as sp
import math

x = sp.Symbol('x')
f_expr = x**2 - 0.5 * sp.exp(-x)
f = sp.lambdify(x, f_expr, modules=["numpy"])

a, b = 0.1, 0.6
n = 10
h = (b - a) / n
x_values = np.linspace(a, b, n + 1)
y_values = f(x_values)

x_star2 = 0.12
x_star3 = 0.58
x_star4 = 0.33

df_values = pd.DataFrame({'x': x_values, 'f(x)': y_values})
print("Таблица значений функции:")
print(df_values.to_string(index=False))
print()


def finite_differences_table(y_values: np.ndarray) -> pd.DataFrame:
    """
    Constructs a finite differences table from the given array of function values

    Args:
        y_values (np.ndarray): function values at interpolation nodes

    Returns:
        pd.DataFrame: Finite differences table
    """
    n = len(y_values)
    table = np.zeros((n, n))
    table[:, 0] = y_values

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

    columns = [f"Δ^{j}f" if j > 0 else "f(x)" for j in range(n)]
    return pd.DataFrame(table, columns=columns)

df_differences = finite_differences_table(y_values)
print("Таблица конечных разностей:")
print(df_differences.round(6).to_string(index=False))
print()


def newton_forward_interpolation(
    x_star: float,
    x_values: np.ndarray,
    diff_table: pd.DataFrame,
    h: float,
    order: int
) -> float:
    """
    Computes the forward Newton interpolation polynomial at a given point

    Args:
        x_star (float): point at which interpolation is performed
        x_values (np.ndarray): array of equally spaced nodes
        diff_table (pd.DataFrame): finite differences table
        h (float): step between nodes
        order (int): interpolation polynomial order

    Returns:
        float: Interpolated value at x_star
    """
    t = (x_star - x_values[0]) / h
    result = diff_table.iloc[0, 0]

    product = 1.0
    for k in range(1, order + 1):
        product *= (t - (k - 1))
        delta = diff_table.iloc[0, k]
        result += (product / math.factorial(k)) * delta

    return result


def newton_backward_interpolation(
    x_star: float,
    x_values: np.ndarray,
    diff_table: pd.DataFrame,
    h: float,
    order: int
) -> float:
    """
     Computes the backward Newton interpolation polynomial at a given point.

     Args:
         x_star (float): point at which interpolation is performed
         x_values (np.ndarray): array of equally spaced nodes
         diff_table (pd.DataFrame): finite differences table
         h (float): step between nodes.
         order (int): interpolation polynomial order

     Returns:
         float: interpolated value at x_star
     """
    n = len(x_values) - 1
    p = (x_star - x_values[n]) / h
    result = diff_table.iloc[n, 0]

    product = 1.0
    for k in range(1, order + 1):
        product *= (p + (k - 1))
        delta = diff_table.iloc[n - k, k]
        result += (product / math.factorial(k)) * delta

    return result


def gauss_forward_interpolation(
    x_star: float,
    x_values: np.ndarray,
    diff_table: pd.DataFrame,
    h: float,
    order: int
) -> float:
    """
    Computes the Gauss forward interpolation polynomial at a given point

    Args:
        x_star (float): point at which interpolation is performed
        x_values (np.ndarray): array of equally spaced nodes
        diff_table (pd.DataFrame): finite differences table
        h (float): step between nodes
        order (int): interpolation polynomial order

    Returns:
        float: interpolated value at x_star
    """
    n = len(x_values)
    mid = n // 2
    x0 = x_values[mid]
    t = (x_star - x0) / h
    result = diff_table.iloc[mid, 0]
    product = 1.0

    for k in range(1, order + 1):
        if k % 2 == 1:
            i = mid - (k // 2)
            product *= (t - ((k - 1) // 2))
        else:
            i = mid - (k // 2)
            product *= (t + (k // 2 - 1))
        result += (product / math.factorial(k)) * diff_table.iloc[i, k]
    return result


def estimate_remainder_bounds(
    f_expr: sp.Expr,
    x_nodes: np.ndarray,
    x_star: float,
    order: int
) -> tuple[float, float]:
    """
       Estimates the bounds for interpolation error using the remainder term of the interpolation polynomial

       Args:
           f_expr (sp.Expr): symbolic expression of the function
           x_nodes (np.ndarray): array of interpolation nodes
           x_star (float): point at which interpolation is performed
           order (int): order of the interpolation polynomial

       Returns:
           tuple[float, float]: (minimum error estimate, maximum error estimate)
       """
    x = sp.Symbol('x')

    deriv = f_expr
    for _ in range(order + 1):
        deriv = sp.diff(deriv, x)

    f_deriv = sp.lambdify(x, deriv, modules=["numpy"])

    x_min = min(np.min(x_nodes), x_star)
    x_max = max(np.max(x_nodes), x_star)
    grid = np.linspace(x_min, x_max, 1000)
    values = np.abs(f_deriv(grid))

    f_min = np.min(values)
    f_max = np.max(values)

    omega = np.prod([x_star - xi for xi in x_nodes])
    omega_abs = abs(omega)

    r_min = (f_min / math.factorial(order + 1)) * omega_abs
    r_max = (f_max / math.factorial(order + 1)) * omega_abs

    return r_min, r_max


# --- Интерполяция в x** = 0.12 по формуле Ньютона вперёд ---
L_x2 = newton_forward_interpolation(x_star2, x_values, df_differences, h, order=4)
f_true = f(x_star2)
error2 = abs(L_x2 - f_true)
r_min2, r_max2 = estimate_remainder_bounds(f_expr, x_values[:5], x_star2, order=4)

print("--- Интерполяция Ньютона вперёд (1-я формула) в точке x** ---")
print(f"f(x**) = {f_true}")
print(f"L(x**) = {L_x2}")
print(f"Абсолютная погрешность: {error2}")
print(f"R_min2 = {r_min2}")
print(f"R_max2 = {r_max2}")
if r_min2 < error2 < r_max2:
    print("min(R) < |L - f| < max(R) выполняется")
else:
    print("min(R) < |L - f| < max(R) не выполняется")

# --- Интерполяция в x*** = 0.58 по формуле Ньютона назад ---
L_x3 = newton_backward_interpolation(x_star3, x_values, df_differences, h, order=4)
f_true3 = f(x_star3)
error3 = abs(L_x3 - f_true3)
r_min3, r_max3 = estimate_remainder_bounds(f_expr, x_values[-5:], x_star3, order=4)

print("\n--- Интерполяция Ньютона назад (2-я формула) в точке x*** ---")
print(f"f(x***) = {f_true3}")
print(f"L(x***) = {L_x3}")
print(f"Абсолютная погрешность: {error3}")
print(f"R_min3 = {r_min3}")
print(f"R_max3 = {r_max3}")
if r_min3 < error3 < r_max3:
    print("min(R) < |L - f| < max(R) выполняется")
else:
    print("min(R) < |L - f| < max(R) не выполняется")

# --- Интерполяция в x**** = 0.33 по формуле Гаусса вперёд ---
L_x4 = gauss_forward_interpolation(x_star4, x_values, df_differences, h, order=4)
f_true4 = f(x_star4)
error4 = abs(L_x4 - f_true4)
mid = len(x_values) // 2
x_subset = x_values[mid - 2:mid + 3]
r_min4, r_max4 = estimate_remainder_bounds(f_expr, x_subset, x_star4, order=4)

print("\n--- Интерполяция Гаусса (1-я формула) в точке x**** ---")
print(f"f(x****) = {f_true4}")
print(f"L(x****) = {L_x4}")
print(f"Абсолютная погрешность: {error4}")
print(f"R_min4 = {r_min4}")
print(f"R_max4 = {r_max4}")
if r_min4 < error4 < r_max4:
    print("min(R) < |L - f| < max(R) не выполняется")
else:
    print("min(R) < |L - f| < max(R) выполняется")
