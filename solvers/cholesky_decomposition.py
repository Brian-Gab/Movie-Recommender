# Solving linear systems using Cholesky decomposition
import numpy as np


def cholesky_decomposition(A, b):
  L = np.linalg.cholesky(A)
  k = len(b)
  y = np.zeros(k)
  for i in range(k):
    y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
  x = np.zeros(k)
  for i in range(k-1, -1, -1):
    x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
  res = float(np.linalg.norm(A @ x - b))
  return x, res