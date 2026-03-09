# Solving linear systems using Cholesky decomposition
# Bare algorithm — no np.linalg.cholesky, no LAPACK calls
import numpy as np


def cholesky_decomposition(A, b):
    """
    Solves Ax = b for SPD matrix A using Cholesky factorization A = L @ L.T,
    implemented entirely from scratch via the Cholesky-Banachiewicz algorithm.

    Steps:
      1. Compute L column-by-column (Cholesky-Banachiewicz)
      2. Forward substitution:  L  @ y = b  →  y
      3. Backward substitution: L.T @ x = y  →  x
    """
    k = len(b)
    L = np.zeros((k, k))

    # ── Step 1: Cholesky-Banachiewicz factorization ───────────────────────
    for j in range(k):
        # Diagonal entry: L[j,j] = sqrt(A[j,j] - sum(L[j,p]^2, p<j))
        s = A[j, j]
        for p in range(j):
            s -= L[j, p] ** 2
        L[j, j] = s ** 0.5

        # Sub-diagonal entries: L[i,j] = (A[i,j] - sum(L[i,p]*L[j,p], p<j)) / L[j,j]
        for i in range(j + 1, k):
            s = A[i, j]
            for p in range(j):
                s -= L[i, p] * L[j, p]
            L[i, j] = s / L[j, j]

    # ── Step 2: Forward substitution  L @ y = b ──────────────────────────
    y = np.zeros(k)
    for i in range(k):
        s = b[i]
        for p in range(i):
            s -= L[i, p] * y[p]
        y[i] = s / L[i, i]

    # ── Step 3: Backward substitution  L.T @ x = y ───────────────────────
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        s = y[i]
        for p in range(i + 1, k):
            s -= L[p, i] * x[p]   # L.T[i,p] == L[p,i]
        x[i] = s / L[i, i]

    res = float(np.sqrt(sum((A[i, :] @ x - b[i]) ** 2 for i in range(k))))
    return x, res
