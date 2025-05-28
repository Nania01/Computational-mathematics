import numpy as np

def thomas_algorithm(A: np.ndarray, B: np.ndarray, C: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Solves a tridiagonal system of linear equations using the Thomas algorithm.

    This function is used to efficiently solve systems of the form:
        A_k * U_{k-1} + B_k * U_k + C_k * U_{k+1} = F_k
    where A, B, and C are the subdiagonal, main diagonal, and superdiagonal of the matrix.

    Args:
        A (np.ndarray): Subdiagonal coefficients (length N), with A[0] unused or set to 0.
        B (np.ndarray): Main diagonal coefficients (length N).
        C (np.ndarray): Superdiagonal coefficients (length N), with C[N-1] unused or set to 0.
        F (np.ndarray): Right-hand side vector (length N).

    Returns:
        np.ndarray: Solution vector U of length N that satisfies the system.
    """
    N = len(B)
    alpha = np.zeros(N)
    beta = np.zeros(N)
    U = np.zeros(N)

    alpha[1] = -C[0] / B[0]
    beta[1] = F[0] / B[0]

    for k in range(1, N - 1):
        denom = B[k] + A[k] * alpha[k]
        alpha[k + 1] = -C[k] / denom
        beta[k + 1] = (F[k] - A[k] * beta[k]) / denom

    U[N - 1] = (F[N - 1] - A[N - 1] * beta[N - 1]) / (A[N - 1] * alpha[N - 1] + B[N - 1])

    for k in range(N - 2, -1, -1):
        U[k] = alpha[k + 1] * U[k + 1] + beta[k + 1]

    return U


A = [0, 1, 1]
B = [4, 4, 4]
C = [1, 1, 0]
F = [5, 5, 5]

U = thomas_algorithm(A, B, C, F)
print("Решение:", U)
