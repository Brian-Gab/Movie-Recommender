# Research Plan: Convergence of Linear System Solvers in ALS-Based Collaborative Filtering

## 1. Research Question

> How does the choice of linear system solver affect the **convergence behaviour**, **numerical stability**, and **computational efficiency** of Alternating Least Squares (ALS) matrix factorization for collaborative filtering on sparse rating data?

**Solvers under study (6):**

| # | Solver | Type | Complexity | Key Property |
|---|--------|------|------------|--------------|
| 1 | Gaussian Elimination (partial pivoting) | Direct | O(k³/3) | Standard baseline; pivot strategy prevents zero-division |
| 2 | Matrix Inverse | Direct | O(k³) | Explicit A⁻¹ amplifies error by κ(A)² |
| 3 | LU Decomposition | Direct | O(k³) + O(k²) solve | Factorization reusable; numerically robust with pivoting |
| 4 | QR Decomposition | Direct | O(k³) | Orthogonal Q has κ(Q)=1; best numerical stability |
| 5 | Gauss-Jacobi | Iterative | O(k²)/iter | Convergence depends on spectral radius; may diverge for small λ |
| 6 | Gauss-Seidel | Iterative | O(k²)/iter | Always converges for SPD; ~2× fewer iters than Jacobi |

---

## 2. Background: ALS for Collaborative Filtering

### 2.1 Problem Setup

Given a sparse rating matrix $R \in \mathbb{R}^{m \times n}$ (users × items), approximate it as a low-rank product:

$$R \approx U V^T, \quad U \in \mathbb{R}^{m \times k}, \quad V \in \mathbb{R}^{n \times k}$$

where $k \ll \min(m, n)$ is the latent rank. Only observed entries (where $R_{ui} > 0$) contribute to the loss:

$$\min_{U, V} \sum_{(u,i) \in \Omega} (R_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \left( \|U\|_F^2 + \|V\|_F^2 \right)$$

where $\Omega$ is the set of observed ratings and $\lambda$ is the L2 regularization parameter.

### 2.2 ALS Algorithm — Step by Step

ALS resolves the joint non-convex optimization by alternately fixing one factor and solving for the other. Each sub-problem becomes a **convex least-squares problem** with a closed-form solution expressed as a **linear system**.

```
Input: R_train (m×n sparse), k (latent rank), λ (regularization), T (epochs)
Output: U (m×k), V (n×k)

1. Initialize U, V randomly ~ N(0, 0.1)
2. Precompute index lists:
     I_u = {items rated by user u}      for all u
     U_i = {users who rated item i}     for all i

3. For epoch = 1, 2, ..., T:

   ┌─ STEP A: Fix V, update each user u ──────────────────────┐
   │  For u = 1, ..., m:                                       │
   │    V_Iu = V[I_u, :]              # sub-matrix of V        │
   │    r_u  = R_train[u, I_u]        # observed ratings       │
   │                                                            │
   │    A = V_Iu^T · V_Iu + λI_k      # k×k, symmetric PD     │
   │    b = V_Iu^T · r_u              # k-vector               │
   │                                                            │
   │    ★ SOLVE: A · x = b  ← THIS IS WHERE SOLVERS PLUG IN   │
   │                                                            │
   │    u_u = x                        # new user factor        │
   └───────────────────────────────────────────────────────────┘

   ┌─ STEP B: Fix U, update each item i ──────────────────────┐
   │  For i = 1, ..., n:                                       │
   │    U_Ui = U[U_i, :]              # sub-matrix of U        │
   │    r_i  = R_train[U_i, i]        # observed ratings       │
   │                                                            │
   │    A = U_Ui^T · U_Ui + λI_k      # k×k, symmetric PD     │
   │    b = U_Ui^T · r_i              # k-vector               │
   │                                                            │
   │    ★ SOLVE: A · x = b  ← SAME SOLVER INTERFACE            │
   │                                                            │
   │    v_i = x                        # new item factor        │
   └───────────────────────────────────────────────────────────┘

   Log metrics: Train RMSE, Test RMSE, Loss, Residual, κ(A), Time
```

### 2.3 The Linear Sub-Problem: Ax = b

Every ALS update solves a **k×k symmetric positive definite** system. The key properties:

- **A = V^T V + λI** is always SPD because V^T V is positive semi-definite and λI makes it strictly positive definite
- **Condition number**: $\kappa(A) = \frac{\sigma_{\max}^2 + \lambda}{\sigma_{\min}^2 + \lambda}$, where $\sigma$ are singular values of V_{I_u}
- **Size**: k×k (typically k = 5–50), so the sub-problem is small but solved **thousands of times** per epoch (once per user + once per item = m+n ≈ 2,625 solves per epoch for ML-100K)

This is the critical point: **the solver is called ~2,625 times per epoch × 50 epochs = ~131,250 total solves**. Small differences in accuracy, stability, or speed compound dramatically.

---

## 3. How Each Solver Interacts with ALS

### 3.1 Gaussian Elimination (with Partial Pivoting)

**Process**: Form augmented matrix [A|b], row-reduce to upper triangular form, back-substitute.

**In ALS context**:
- Partial pivoting swaps rows to place the largest absolute value on the diagonal, preventing division by near-zero pivots
- For SPD matrices, pivoting is rarely needed (diagonals are always positive), but it guards against numerical edge cases
- Each solve is independent — no factorization reuse between users/items

**Residual**: Machine precision (~10⁻¹³)

### 3.2 Matrix Inverse

**Process**: Compute x = A⁻¹b explicitly.

**In ALS context**:
- Forming A⁻¹ explicitly is numerically wasteful — it squares the condition number effect
- Error bound: $\|x - \hat{x}\| / \|x\| \leq \kappa(A)^2 \cdot \epsilon_{\text{machine}}$
- Still works well in practice for small k and moderate κ(A), but is the **least stable** direct method
- Each A matrix is different (different subset of items/users), so the inverse cannot be cached

### 3.3 LU Decomposition

**Process**: Factor A = PLU (P = permutation, L = lower triangular, U = upper triangular), then solve Ly = Pb (forward substitution) and Ux = y (back substitution).

**In ALS context**:
- Most efficient direct method when the same A is used for multiple right-hand sides (not the case here — each user has a different A)
- Numerically equivalent to Gaussian elimination but structured as a factorization
- Shares the same O(k³) cost but with smaller constants due to structured access patterns

### 3.4 QR Decomposition

**Process**: Factor A = QR (Q orthogonal, R upper triangular), then solve Rx = Q^T b.

**In ALS context**:
- **Best numerical stability** among all methods tested
- Since Q is orthogonal, κ(Q) = 1, so the condition number is not amplified during the solve
- The residual ‖Ax - b‖ is minimized in the least-squares sense
- Slightly more expensive than LU (constant factor ~2×), but the stability gain is significant when κ(A) is large

### 3.5 Gauss-Jacobi Iteration

**Process**: Split A = D + R_off. Iterate: $x^{(t+1)} = D^{-1}(b - R_{\text{off}} x^{(t)})$

**In ALS context**:
- Convergence requires spectral radius $\rho(D^{-1} R_{\text{off}}) < 1$
- For A = V^T V + λI: λ controls diagonal dominance. **Small λ → weak diagonal → slow/no convergence**
- Each inner iteration costs O(k²) vs O(k³) for direct methods — beneficial when k is large and few iterations suffice
- **Does NOT always converge** for ALS sub-problems (this is a key finding)
- Returns an approximate solution, introducing a small error that accumulates across ALS epochs

### 3.6 Gauss-Seidel Iteration

**Process**: Like Jacobi but uses updated components immediately (in-place updates within each iteration).

**In ALS context**:
- **Always converges for SPD matrices** (guaranteed by theory, and A in ALS is always SPD)
- Convergence rate: $\rho(T_{GS}) = \rho(T_J)^2$ — roughly **half the iterations** of Jacobi
- Same O(k²) per-iteration cost as Jacobi
- More sequential (cannot parallelize within one iteration), but the faster convergence compensates

---

## 4. Experimental Pipeline

### 4.1 Dataset

**MovieLens 100K** (GroupLens):
- 943 users, 1,682 items, 100,000 ratings (1–5 scale)
- Sparsity: ~94%
- Train/test split: 90%/10% of observed ratings (stratified random)

### 4.2 Metrics Logged Per Epoch

| Metric | Formula | Purpose |
|--------|---------|---------|
| Train RMSE | $\sqrt{\frac{1}{|\Omega_{\text{train}}|} \sum (R_{ui} - \hat{R}_{ui})^2}$ | Measures fit to training data |
| Test RMSE | Same formula on held-out set | Generalization accuracy |
| Regularized Loss | $\sum (R-UV^T)^2 + \lambda(\|U\|^2 + \|V\|^2)$ | Actual ALS objective |
| Mean Residual ‖Ax-b‖ | Avg over all sub-problems in epoch | Sub-problem solve accuracy |
| Mean Inner Iterations | Avg iters (iterative solvers only) | Iterative solver effort |
| Mean Condition Number κ(A) | Avg $\kappa(A)$ over all sub-problems | Numerical difficulty |
| Cumulative Wall-Clock Time | Seconds since start | Computational cost |

### 4.3 Experiments

#### Experiment 1 — Baseline Convergence (all 6 solvers)

- **Setup**: k=10, λ=0.1, 50 epochs, seed=42
- **Purpose**: Establish baseline convergence curves for all solvers under typical conditions
- **Plots**: 6-panel dashboard (Train RMSE, Test RMSE, Loss, Residual, |ΔRMSE|, Time)
- **Key questions**:
  - Do all solvers converge to the same RMSE?
  - Which solver converges fastest (fewest epochs)?
  - What is the residual profile for direct vs iterative solvers?

#### Experiment 2 — Regularization λ Sensitivity

- **Setup**: λ ∈ {0.001, 0.01, 0.1, 0.5, 1.0}, k=10, 30 epochs
- **Purpose**: Study how λ affects condition number and solver stability
- **Plots**: Test RMSE vs λ, Mean κ(A) vs λ, convergence curves at extreme λ
- **Key questions**:
  - At what λ does Gauss-Jacobi fail to converge?
  - How does κ(A) scale with λ?
  - Is there a λ sweet spot balancing accuracy and stability?

#### Experiment 3 — Latent Rank k Sensitivity

- **Setup**: k ∈ {5, 7, 10, 20}, λ=0.1, 30 epochs
- **Purpose**: Study how sub-problem size affects solver performance
- **Plots**: Test RMSE vs k, Time vs k, time scaling analysis (O(k³) vs O(k²) guides)
- **Key questions**:
  - Do iterative solvers scale better than direct solvers as k grows?
  - Does larger k worsen conditioning?
  - At what k do iterative solvers become competitive in wall-clock time?

### 4.4 Hypotheses

| ID | Hypothesis | Rationale |
|----|-----------|-----------|
| H1 | All solvers converge to the same final RMSE | They all target the same ALS fixed point; only the path differs |
| H2 | Direct solvers converge in fewer ALS epochs | Exact sub-solves give cleaner descent directions |
| H3 | QR achieves the lowest sub-problem residual | Orthogonal factorization avoids κ(A) amplification |
| H4 | Gauss-Seidel needs ~half the inner iterations of Jacobi | $\rho(T_{GS}) = \rho(T_J)^2$ for SPD matrices |
| H5 | Small λ destabilizes Gauss-Jacobi | λI controls diagonal dominance; small λ → large spectral radius |
| H6 | Iterative solvers scale better with k | O(k²)/iter vs O(k³) for direct methods |

---

## 5. Deliverables

### 5.1 Notebook (`als_convergence_study.ipynb`)

Fully executable notebook with:
- Data loading and EDA
- All 6 solver implementations with unified interface
- ALS engine with convergence logging
- 3 experiments with plots and summary tables
- Per-solver CSV logs in `outputs/`

### 5.2 Research Paper

**Suggested Outline:**

1. **Introduction** — Motivation: collaborative filtering, matrix factorization, ALS. Why the sub-problem solver matters.
2. **Related Work** — ALS in recommender systems, numerical linear algebra fundamentals, prior comparisons of solvers.
3. **Methodology**
   - ALS algorithm description
   - The 6 linear system solvers (theory, complexity, convergence guarantees)
   - How each solver integrates with ALS (Section 3 of this plan)
   - Experimental setup (dataset, metrics, hyperparameters)
4. **Results and Discussion**
   - Experiment 1: Baseline convergence comparison
   - Experiment 2: Regularization sensitivity analysis
   - Experiment 3: Rank sensitivity and scaling
   - Hypothesis evaluation
5. **Conclusion** — Summary of findings, practical recommendations for solver selection in ALS.
6. **References**

---

## 6. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data loading & EDA | ✅ Done | ML-100K, 90/10 split |
| Gaussian Elimination | ✅ Done | With partial pivoting |
| Matrix Inverse | ✅ Done | Explicit A⁻¹b |
| LU Decomposition | ✅ Done | scipy lu_factor/lu_solve |
| QR Decomposition | ✅ Done | Manual back-substitution |
| Gauss-Jacobi | ✅ Done | tol=1e-8, max_iter=500 |
| Gauss-Seidel | ✅ Done | tol=1e-8, max_iter=500 |
| ALS engine | ✅ Done | Pluggable solver, 7-metric logging |
| Experiment 1 | ⚠️ Partial | Code done; CSVs exist for 5 solvers (missing Jacobi, Seidel); needs re-run |
| Experiment 2 | ❌ Not run | Code ready |
| Experiment 3 | ❌ Not run | Code ready |
| Paper | ❌ Not started | — |

**Next steps**: Remove Cramer's Rule from notebook, re-run all experiments with 6 solvers, generate plots, write paper.