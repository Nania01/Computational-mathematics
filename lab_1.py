import numpy as np
import pandas as pd
import sympy as sp
import math

x = sp.Symbol('x')
f_expr = x**2 - 0.5 * sp.exp(-x)
f = sp.lambdify(x, f_expr, modules=['numpy'])

a, b = 0.1, 0.6
n = 10
h = (b - a) / n
x_values = np.linspace(a, b, n + 1)
y_values = f(x_values)

x_star = 0.37

df = pd.DataFrame({'x': x_values, 'f(x)': y_values})
print(df)


def lagrange_interpolation(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_star: float,
    order: int
) -> tuple[float, np.ndarray]:
    """
    Computes the Lagrange interpolation polynomial of given order at point x_star.

    Args:
        x_nodes (np.ndarray): array of interpolation nodes
        y_nodes (np.ndarray): function values at the interpolation nodes
        x_star (float): the point at which to interpolate
        order (int): the order of the interpolation polynomial

    Returns:
        tuple[float, np.ndarray]:
            - interpolated value at x_star
            - array of nodes used for interpolation
    """
    if order >= len(x_nodes):
        raise ValueError("Порядок интерполяции выше количества доступных точек")

    distances = np.abs(x_nodes - x_star)
    indices = np.argsort(distances)[:order + 1]
    x_selected = x_nodes[indices]
    y_selected = y_nodes[indices]

    result = 0.0
    for j in range(order + 1):
        term = y_selected[j]
        for m in range(order + 1):
            if m != j:
                term *= (x_star - x_selected[m]) / (x_selected[j] - x_selected[m])
        result += term

    return result, x_selected


def lagrange_error_bounds(
    f_expr: sp.Expr,
    x_nodes: np.ndarray,
    x_star: float,
    order: int
) -> tuple[float, float]:
    """
    Estimates the bounds for interpolation error of Lagrange polynomial.

    Args:
        f_expr (sp.Expr): symbolic expression of the function
        x_nodes (np.ndarray): array of interpolation nodes
        x_star (float): point at which interpolation is performed
        order (int): order of the interpolation polynomial

    Returns:
        tuple[float, float]: (minimum estimate, maximum estimate) of the interpolation error
    """
    x = sp.Symbol('x')
    deriv = f_expr
    for _ in range(order + 1):
        deriv = sp.diff(deriv, x)
    f_deriv = sp.lambdify(x, deriv, modules=["numpy"])

    x_min = min(np.min(x_nodes), x_star)
    x_max = max(np.max(x_nodes), x_star)

    xx = np.linspace(x_min, x_max, 1000)
    deriv_values = np.abs(f_deriv(xx))
    min_deriv = np.min(deriv_values)
    max_deriv = np.max(deriv_values)

    product = np.prod([x_star - xi for xi in x_nodes])
    r_min = (min_deriv / math.factorial(order + 1)) * abs(product)
    r_max = (max_deriv / math.factorial(order + 1)) * abs(product)

    return r_min, r_max


def divided_differences(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray
) -> np.ndarray:
    """
    Computes coefficients of Newton's divided differences table.

    Args:
        x_nodes (np.ndarray): array of interpolation nodes
        y_nodes (np.ndarray): function values at the nodes

    Returns:
        np.ndarray: array of divided difference coefficients
    """
    n = len(x_nodes)
    coef = np.copy(y_nodes).astype(float)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x_nodes[j:n] - x_nodes[0:n - j])

    return coef


def newton_interpolation(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_star: float,
    order: int
) -> tuple[float, np.ndarray]:
    """
    Computes the Newton interpolation polynomial of given order at point x_star.

    Args:
        x_nodes (np.ndarray): array of interpolation nodes
        y_nodes (np.ndarray): function values at the interpolation nodes
        x_star (float): point at which to evaluate the interpolant
        order (int): order of the interpolation polynomial

    Returns:
        tuple[float, np.ndarray]:
            - interpolated value at x_star
            - array of nodes used for interpolation
    """
    if order >= len(x_nodes):
        raise ValueError("Порядок интерполяции выше количества доступных точек")

    distances = np.abs(x_nodes - x_star)
    indices = np.argsort(distances)[:order + 1]

    x_selected = x_nodes[indices]
    y_selected = y_nodes[indices]

    sorted_indices = np.argsort(x_selected)
    x_selected = x_selected[sorted_indices]
    y_selected = y_selected[sorted_indices]

    coefs = divided_differences(x_selected, y_selected)

    result = coefs[0]
    product = 1.0
    for i in range(1, order + 1):
        product *= (x_star - x_selected[i - 1])
        result += coefs[i] * product

    return result, x_selected


# -------------------- Лагранж 1 порядка --------------------
lagrange_value, x_used = lagrange_interpolation(x_values, y_values, x_star, order=1)
f_exact = f(x_star)
real_error = abs(lagrange_value - f_exact)
r_min, r_max = lagrange_error_bounds(f_expr, x_used, x_star, order=1)

print("\n--- Интерполяция Лагранжа 1-го порядка ---")
print("Истинное значение f(x*):", f_exact)
print(f"L_1({x_star}) =", lagrange_value)
print(f"Абсолютная погрешность: |L_1 - f(x*)| = {real_error}")

if real_error <= 0.0001:
    print("Interpolation is ACCEPTABLE (|L_1 - f(x*)| ≤ 0.0001)")
else:
    print("Interpolation is NOT acceptable (|L_1 - f(x*)| > 0.0001)")

print(f"R1_min = {r_min}")
print(f"R1_max = {r_max}")

if r_min < real_error < r_max:
    print("Actual error is WITHIN the theoretical remainder bounds")
else:
    print("Actual error is OUTSIDE the theoretical remainder bounds")


# -------------------- Лагранж 2 порядка --------------------
lagrange_value_2, x_used_2 = lagrange_interpolation(x_values, y_values, x_star, order=2)
real_error_2 = abs(lagrange_value_2 - f_exact)
r2_min, r2_max = lagrange_error_bounds(f_expr, x_used_2, x_star, order=2)

print("\n--- Интерполяция Лагранжа 2-го порядка ---")
print(f"L_2({x_star}) =", lagrange_value_2)
print(f"Абсолютная погрешность: |L_2 - f(x*)| = {real_error_2}")

if real_error_2 <= 0.00001:
    print("Interpolation is ACCEPTABLE (|L_2 - f(x*)| ≤ 0.00001)")
else:
    print("Interpolation is NOT acceptable (|L_2 - f(x*)| > 0.00001)")

print(f"R2_min = {r2_min}")
print(f"R2_max = {r2_max}")

if r2_min < real_error_2 < r2_max:
    print("Actual error is WITHIN the theoretical remainder bounds")
else:
    print("Actual error is OUTSIDE the theoretical remainder bounds")


# -------------------- Ньютон 1 порядка --------------------
newton_val_1, newton_nodes_1 = newton_interpolation(x_values, y_values, x_star, order=1)
error_n1 = abs(newton_val_1 - f_exact)

print("\n--- Интерполяция Ньютона 1-го порядка ---")
print(f"N_1({x_star}) =", newton_val_1)
print(f"Абсолютная погрешность: |N_1 - f(x*)| = {error_n1}")


# -------------------- Ньютон 2 порядка --------------------
newton_val_2, newton_nodes_2 = newton_interpolation(x_values, y_values, x_star, order=2)
error_n2 = abs(newton_val_2 - f_exact)

print("\n--- Интерполяция Ньютона 2-го порядка ---")
print(f"N_2({x_star}) =", newton_val_2)
print(f"Абсолютная погрешность: |N_2 - f(x*)| = {error_n2}")


# -------------------- Сравнение Лагранжа и Ньютона --------------------
print("\n--- Разности между значениями Лагранжа и Ньютона ---")
print(f"|L_1 - N_1| = {abs(lagrange_value - newton_val_1)}")
print(f"|L_2 - N_2| = {abs(lagrange_value_2 - newton_val_2)}")
