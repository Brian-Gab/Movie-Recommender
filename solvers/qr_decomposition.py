# Solving linear systems using QR decomposition
# Bare algorithm — no np.linalg.qr, no LAPACK calls
import numpy as np


def qr_decomposition(A, b):
    """
    Solves Ax = b via QR factorization A = Q @ R,
    implemented entirely from scratch using Modified Gram-Schmidt orthogonalization.

    Steps:
      1. Modified Gram-Schmidt → Q (orthonormal), R (upper triangular)
      2. Compute b_hat = Q.T @ b  (using explicit dot products)
      3. Back substitution: R @ x = b_hat  →  x
    """
    k = len(b)
    Q = np.zeros((k, k))
    R = np.zeros((k, k))

    # ── Step 1: Modified Gram-Schmidt QR ─────────────────────────────────
    # Work on columns of A
    V = A.astype(float).copy()   # V[:,j] will be modified in place

    for j in range(k):
        # R[j,j] = ||v_j||
        norm = 0.0
        for i in range(k):
            norm += V[i, j] ** 2
        R[j, j] = norm ** 0.5

        # Q[:,j] = v_j / ||v_j||
        for i in range(k):
            Q[i, j] = V[i, j] / R[j, j]

        # Subtract projection from all subsequent columns (modified GS)
        for l in range(j + 1, k):
            # R[j,l] = Q[:,j] . V[:,l]
            dot = 0.0
            for i in range(k):
                dot += Q[i, j] * V[i, l]
            R[j, l] = dot
            # V[:,l] -= R[j,l] * Q[:,j]
            for i in range(k):
                V[i, l] -= R[j, l] * Q[i, j]

    # ── Step 2: b_hat = Q.T @ b ───────────────────────────────────────────
    b_hat = np.zeros(k)
    for i in range(k):
        s = 0.0
        for r in range(k):
            s += Q[r, i] * b[r]   # Q.T[i,r] = Q[r,i]
        b_hat[i] = s

    # ── Step 3: Back substitution  R @ x = b_hat ─────────────────────────
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        s = b_hat[i]
        for p in range(i + 1, k):
            s -= R[i, p] * x[p]
        x[i] = s / R[i, i]

    res = float(np.sqrt(sum((A[i, :] @ x - b[i]) ** 2 for i in range(k))))
    return x, res
