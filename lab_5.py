import numpy as np
import matplotlib.pyplot as plt


def compute_slopes(xvals: list[float], yvals: list[float], f2_a: float, f2_b: float) -> np.ndarray:
    """
    Computes the slope values m_i at spline nodes using the method of slopes
    with second-type boundary conditions (specified second derivatives at endpoints).

    This function constructs and solves a linear system to determine the first derivatives (slopes)
    at the interpolation nodes, based on the function values and second derivatives at the boundaries.

    Args:
        xvals (list[float]): List of x-coordinates of interpolation nodes (length n + 1).
        yvals (list[float]): List of y-coordinates of interpolation nodes (length n + 1).
        f2_a (float): Second derivative at the left endpoint (x = x_0).
        f2_b (float): Second derivative at the right endpoint (x = x_n).

    Returns:
        np.ndarray: Array of slope values m_i (first derivatives) at the nodes.
    """
    x = np.array(xvals)
    y = np.array(yvals)
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 2
    A[0, 1] = 1
    B[0] = 3 * (y[1] - y[0]) / h[0] - (h[0] * f2_a) / 2

    for i in range(1, n):
        mu = h[i - 1] / (h[i - 1] + h[i])
        lmb = h[i] / (h[i - 1] + h[i])
        A[i, i - 1] = mu
        A[i, i] = 2
        A[i, i + 1] = lmb
        B[i] = 3 * (lmb * (y[i + 1] - y[i]) / h[i] + mu * (y[i] - y[i - 1]) / h[i - 1])

    A[n, n - 1] = 1
    A[n, n] = 2
    B[n] = 3 * (y[n] - y[n - 1]) / h[n - 1] - (h[n - 1] * f2_b) / 2

    m = np.linalg.solve(A, B)

    return m


def spline_value(x: float, xvals: list[float], yvals: list[float], m: np.ndarray) -> float:
    """
    Evaluates the spline function S(x) at a given point x.

    Args:
        x (float): The point at which to evaluate the spline.
        xvals (list[float]): x-coordinates of interpolation nodes.
        yvals (list[float]): y-coordinates of interpolation nodes.
        m (np.ndarray): Slopes (first derivatives) at the nodes, computed via compute_slopes().

    Returns:
        float: Value of the spline at point x.
    """
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    h = np.diff(xvals)

    i = np.searchsorted(xvals, x) - 1
    i = np.clip(i, 0, len(h) - 1)
    dx = x - xvals[i]

    a_i = (6 / h[i]) * ((yvals[i + 1] - yvals[i]) / h[i] - (m[i + 1] + 2 * m[i]) / 3)
    b_i = (12 / h[i] ** 2) * ((m[i + 1] + m[i]) / 2 - (yvals[i + 1] - yvals[i]) / h[i])

    return yvals[i] + m[i] * dx + (a_i * dx ** 2) / 2 + (b_i * dx ** 3) / 6


def spline_derivative(x: float, xvals: list[float], yvals: list[float], m: np.ndarray) -> float:
    """
    Evaluates the second derivative S''(x) of the spline at a given point x.

    Args:
        x (float): The point at which to evaluate the second derivative.
        xvals (list[float]): x-coordinates of interpolation nodes.
        yvals (list[float]): y-coordinates of interpolation nodes.
        m (np.ndarray): Slopes (first derivatives) at the nodes, computed via compute_slopes().

    Returns:
        float: Second derivative of the spline at point x.
    """
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    h = np.diff(xvals)

    i = np.searchsorted(xvals, x) - 1
    i = np.clip(i, 0, len(h) - 1)
    dx = x - xvals[i]

    a_i = (6 / h[i]) * ((yvals[i + 1] - yvals[i]) / h[i] - (m[i + 1] + 2 * m[i]) / 3)
    b_i = (12 / h[i] ** 2) * ((m[i + 1] + m[i]) / 2 - (yvals[i + 1] - yvals[i]) / h[i])

    return a_i + b_i * dx


def plot_spline(xvals, yvals, m, f_true=None):
    xs = np.linspace(xvals[0], xvals[-1], 500)
    ys = [spline_value(x, xvals, yvals, m) for x in xs]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="Cubic Spline", color="blue")
    plt.plot(xvals, yvals, 'o', label="Data Points", color="black")
    if f_true is not None:
        plt.plot(xs, [f_true(x) for x in xs], label="True Function", linestyle='dashed', color="red")
    plt.legend()
    plt.grid(True)
    plt.title("Cubic Spline Interpolation")
    plt.show()


xvals = np.linspace(0.1, 0.6, 6)
f = lambda x: 2 * x - np.cos(x)
f2 = lambda x: np.cos(x)

yvals = f(xvals)
m = compute_slopes(xvals, yvals, f2(xvals[0]), -f2(xvals[-1]))

plot_spline(xvals, yvals, m, f_true=f)

print("S''(x_0) =", spline_derivative(xvals[0], xvals, yvals, m))
print("f''(x_0) =", f2(xvals[0]))
print("S''(x_N) =", spline_derivative(xvals[-1], xvals, yvals, m))
print("f''(x_N) =", f2(xvals[-1]))
