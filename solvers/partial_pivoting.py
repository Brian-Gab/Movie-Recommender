# Partial pivoting solver (Gaussian Elimination with Partial Pivoting)
# Bare algorithm — pure Python loops, no LAPACK calls
import numpy as np


def partial_pivoting(A, b):
    """
    Solves Ax = b via Gaussian Elimination with Partial Pivoting,
    implemented entirely from scratch.

    Steps:
      1. Forward elimination with partial pivoting → upper triangular form
      2. Back substitution → x
    """
    k = len(b)
    # Augmented matrix [A | b]
    M = np.zeros((k, k + 1))
    for i in range(k):
        for j in range(k):
            M[i, j] = float(A[i, j])
        M[i, k] = float(b[i])

    # ── Step 1: Forward elimination with partial pivoting ─────────────────
    for col in range(k):
        # Find pivot
        max_val = abs(M[col, col])
        max_row = col
        for row in range(col + 1, k):
            if abs(M[row, col]) > max_val:
                max_val = abs(M[row, col])
                max_row = row

        # Swap rows
        if max_row != col:
            for c in range(k + 1):
                M[col, c], M[max_row, c] = M[max_row, c], M[col, c]

        # Eliminate below pivot
        pivot = M[col, col]
        for row in range(col + 1, k):
            factor = M[row, col] / pivot
            M[row, col] = 0.0     # explicitly zero (avoids floating drift)
            for c in range(col + 1, k + 1):
                M[row, c] -= factor * M[col, c]

    # ── Step 2: Back substitution ─────────────────────────────────────────
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        s = M[i, k]
        for p in range(i + 1, k):
            s -= M[i, p] * x[p]
        x[i] = s / M[i, i]

    # Residual computed once, after back substitution (bug fix vs original)
    res = float(np.sqrt(sum((A[i, :] @ x - b[i]) ** 2 for i in range(k))))
    return x, res
