# Solving linear systems using QR decomposition
# Bare algorithm — no np.linalg.qr, no LAPACK calls
import numpy as np


def qr_decomposition(A, b):
    """
    Solves Ax = b via QR factorization using Householder reflections.

    Steps:
      1. Apply k Householder reflections H_1, ..., H_k to reduce A → R
         (upper triangular), accumulating the same transforms on b → b_hat.
         Each reflection H = I - 2*v*v^T zeros out the sub-diagonal of column j.
         The sign of v is chosen to maximise |v[0]| and avoid cancellation.
      2. Back substitution: R @ x = b_hat  →  x
    """
    k = len(b)
    R = A.astype(float).copy()
    b_hat = b.astype(float).copy()

    # ── Step 1: Householder reflections ───────────────────────────────────
    for j in range(k):
        x = R[j:, j].copy()

        # v = x + sign(x[0]) * ||x|| * e1  (sign chosen to avoid cancellation)
        sign = 1.0 if x[0] >= 0.0 else -1.0
        x[0] += sign * np.sqrt(x @ x)
        norm_v = np.sqrt(x @ x)
        if norm_v == 0.0:
            continue
        v = x / norm_v   # unit Householder vector

        # Apply H = I - 2*v*v^T to the active submatrix and to b_hat
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        b_hat[j:] -= 2.0 * v * (v @ b_hat[j:])

    # ── Step 2: Back substitution  R @ x = b_hat ─────────────────────────
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        x[i] = (b_hat[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]

    res = float(np.sqrt(np.sum((A @ x - b) ** 2)))
    return x, res