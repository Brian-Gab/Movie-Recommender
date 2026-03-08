# Partial pivoting solver

import numpy as np

def partial_pivoting(A, b):
  k = len(b)
  M = np.hstack([A.astype(float).copy(), 
                    b.reshape(-1, 1).astype(float)])
    
  for col in range(k):
        # Partial pivoting
        max_row = col + np.argmax(np.abs(M[col:, col]))
        M[[col, max_row]] = M[[max_row, col]]

        pivot = M[col, col]
        for row in range(col + 1, k):
            factor = M[row, col] / pivot
            M[row,col:] -= factor * M[col, col:]

      # Back substitution
  x = np.zeros(k)
  for i in range(k - 1, -1, -1):
        x[i] = (M[i,k] - np.dot(M[i, i + 1:k], x[i + 1:k])) / M[i, i]
        res = float(np.linalg.norm(A @ x - b))
  return x, res