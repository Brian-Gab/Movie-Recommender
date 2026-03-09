# LU decomposition solver
# Bare algorithm — no scipy.linalg, no LAPACK calls
import numpy as np


def lu_decomposition(A, b):
    """
    Solves Ax = b via LU decomposition with partial pivoting: PA = LU,
    implemented entirely from scratch (Doolittle algorithm).

    Steps:
      1. Doolittle LU factorization with partial pivoting → P, L, U
      2. Apply permutation:       Pb = P @ b
      3. Forward substitution:    L  @ y = Pb  →  y
      4. Backward substitution:   U  @ x = y   →  x
    """
    k = len(b)
    L = np.eye(k)
    U = A.astype(float).copy()
    P = np.eye(k)             # permutation matrix (tracked as integer indices)
    piv = list(range(k))      # pivot index tracker

    # ── Step 1: Doolittle LU with partial pivoting ────────────────────────
    for col in range(k):
        # Find pivot row
        max_val = abs(U[col, col])
        max_row = col
        for row in range(col + 1, k):
            if abs(U[row, col]) > max_val:
                max_val = abs(U[row, col])
                max_row = row

        # Swap rows in U, P, and the already-computed columns of L
        if max_row != col:
            U[[col, max_row]]    = U[[max_row, col]]
            P[[col, max_row]]    = P[[max_row, col]]
            # Swap only the already-filled part of L (columns 0..col-1)
            L[[col, max_row], :col] = L[[max_row, col], :col]

        # Elimination
        for row in range(col + 1, k):
            if U[col, col] == 0.0:
                continue
            factor = U[row, col] / U[col, col]
            L[row, col] = factor
            for c in range(col, k):
                U[row, c] -= factor * U[col, c]

    # ── Step 2: Apply permutation to b ───────────────────────────────────
    pb = P @ b

    # ── Step 3: Forward substitution  L @ y = Pb ─────────────────────────
    y = np.zeros(k)
    for i in range(k):
        s = pb[i]
        for p in range(i):
            s -= L[i, p] * y[p]
        y[i] = s   # L diagonal is all 1s (unit lower triangular)

    # ── Step 4: Backward substitution  U @ x = y ─────────────────────────
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        s = y[i]
        for p in range(i + 1, k):
            s -= U[i, p] * x[p]
        x[i] = s / U[i, i]

    res = float(np.sqrt(sum((A[i, :] @ x - b[i]) ** 2 for i in range(k))))
    return x, res
