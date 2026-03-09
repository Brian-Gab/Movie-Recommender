# Solving Linear Systems using Matrix Inverse
# Bare algorithm — no np.linalg.inv, no LAPACK calls
import numpy as np


def matrix_inverse(A, b):
    """
    Solves Ax = b by explicitly computing A^{-1} via Gauss-Jordan elimination,
    then computing x = A^{-1} @ b. Everything implemented from scratch.

    Steps:
      1. Gauss-Jordan elimination on [A | I] → [I | A^{-1}]
      2. x = A_inv @ b
    """
    k = len(b)

    # ── Step 1: Gauss-Jordan on augmented matrix [A | I] ─────────────────
    # Build augmented matrix: left half = A, right half = identity
    aug = np.zeros((k, 2 * k))
    for i in range(k):
        for j in range(k):
            aug[i, j] = float(A[i, j])
        aug[i, k + i] = 1.0

    for col in range(k):
        # Partial pivoting
        max_val = abs(aug[col, col])
        max_row = col
        for row in range(col + 1, k):
            if abs(aug[row, col]) > max_val:
                max_val = abs(aug[row, col])
                max_row = row
        aug[[col, max_row]] = aug[[max_row, col]]

        # Scale pivot row so diagonal becomes 1
        pivot = aug[col, col]
        for c in range(2 * k):
            aug[col, c] /= pivot

        # Eliminate all other rows (both above and below — Gauss-Jordan)
        for row in range(k):
            if row == col:
                continue
            factor = aug[row, col]
            for c in range(2 * k):
                aug[row, c] -= factor * aug[col, c]

    # Right half of aug is now A^{-1}
    A_inv = aug[:, k:]

    # ── Step 2: x = A^{-1} @ b ───────────────────────────────────────────
    x = np.zeros(k)
    for i in range(k):
        s = 0.0
        for j in range(k):
            s += A_inv[i, j] * b[j]
        x[i] = s

    res = float(np.sqrt(sum((A[i, :] @ x - b[i]) ** 2 for i in range(k))))
    return x, res
