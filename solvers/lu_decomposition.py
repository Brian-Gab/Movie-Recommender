# LU decomposition solver
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

def lu_decomposition(A, b):
  P, L, U = lu(A)

  lu_piv = lu_factor(A)
  x = lu_solve(lu_piv, b)
  res = float(np.linalg.norm(A @ x - b))
  return x, res 