# Solving linear systems using QR decomposition
import numpy as np

def qr_decomposition(A, b):
  Q, R = np.linalg.qr(A)
  b_hat = Q.T @ b
  k = len(b)
  x = np.zeros(k)
  for i in range(k-1, -1, -1):
    x[i] = (b_hat[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
  res = float(np.linalg.norm(A @ x - b))
  return x, res