# Solving Linear Systems using Matrix Inverse
import numpy as np

def matrix_inverse(A, b):
  A_inv = np.linalg.inv(A)
  x = A_inv @ b
  res = np.linalg.norm(A @ x - b)
  return x, res