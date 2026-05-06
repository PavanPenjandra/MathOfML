# Calculus for ML/DL/Generative AI: Zero to Hero (Part 3)

## Part 3: Vector Calculus – The Language of Neural Flows

**File:** `calculus3_vector_calculus.md`

---

# Calculus III: Vector Calculus for Machine Learning

**From Layman to Data Scientist**  
*Analogies • Memory Techniques • Hands-On Examples*

---

## Table of Contents

1. [Vector-Valued Functions & Vector Fields](#vector-fields)
2. [The Jacobian Matrix – Derivative for Vector Functions](#jacobian)
3. [The Hessian – Second Derivatives in High Dimensions](#hessian)
4. [Divergence & Curl – Flows and Rotations](#divcurl)
5. [Line Integrals – Accumulation Along Paths](#line-integrals)
6. [Backpropagation as Vector Chain Rule](#backprop-vector)
7. [Applications in Generative AI: Score Functions & Diffusion](#generative-vector)
8. [Applications in Agentic AI: Policy Gradients & Q-Learning](#agentic-vector)
9. [Matrix Calculus – Derivatives with Respect to Matrices](#matrix-calculus)
10. [Final Project: Implement Autodiff](#final-project-vector)

---

## Module 1: Vector-Valued Functions & Vector Fields <a name="vector-fields"></a>

### 1.1 Vector-Valued Functions
`f: ℝⁿ → ℝᵐ` – maps n inputs to m outputs.  
**Example:** Neural network layer: `f(x) = σ(Wx + b)` (maps ℝᵈ_input to ℝᵈ_output).

### 1.2 Vector Fields
Assign a vector to every point in space.  
**Analogy:** Wind map – at each location, arrow shows wind direction and strength.

**In AI:**
- **Gradient field** of loss function: vector field pointing uphill.
- **Score function** in diffusion models: `∇ log p_t(x)` – vector field pointing toward high probability.

```python
# Visualizing gradient field of f(x,y)=x²-y²
x = np.linspace(-2,2,20)
y = np.linspace(-2,2,20)
X,Y = np.meshgrid(x,y)
U = 2*X   # ∂f/∂x
V = -2*Y  # ∂f/∂y
plt.quiver(X,Y,U,V)
```

---

## Module 2: The Jacobian Matrix – Derivative for Vector Functions <a name="jacobian"></a>

### 2.1 Definition
For `f: ℝⁿ → ℝᵐ`, Jacobian J is m×n matrix:  
`J_ij = ∂f_i / ∂x_j`

Each row is gradient of one output component. Each column shows how all outputs change with one input.

**Analogy:** A sensitivity matrix. If you have multiple sensors (outputs) and multiple controls (inputs), J tells you how each control affects each sensor.

### 2.2 Geometric Meaning
Jacobian linearizes f near a point: `f(x+Δx) ≈ f(x) + J·Δx`.  
- Determinant (square J) = local volume scaling factor.
- Singular values = stretching factors along principal axes.

### 2.3 Jacobian in Neural Networks
- **Forward pass:** `z = Wx + b`, `a = σ(z)`.  
  Jacobian of layer w.r.t input: `J = diag(σ'(z)) * W` (chain rule).
- **Backward pass:** Gradients propagate via Jacobian transpose: `∂L/∂x = Jᵀ (∂L/∂a)`.

### 2.4 Computing with PyTorch Autograd
```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2
J = torch.autograd.functional.jacobian(lambda x: x**2, x)
# J = [[2,0],[0,4]] diagonal
```

### 📝 Exercise
Compute Jacobian of `f(x,y) = [x²y, sin(xy)]` at (1,0).  
Answer:  
∂f₁/∂x=2xy=0, ∂f₁/∂y=x²=1, ∂f₂/∂x=y cos(xy)=0, ∂f₂/∂y=x cos(xy)=1.  
Jacobian = [[0,1],[0,1]]

---

## Module 3: The Hessian – Second Derivatives in High Dimensions <a name="hessian"></a>

### 3.1 Definition
For scalar function `f: ℝⁿ → ℝ`, Hessian H is n×n matrix:  
`H_ij = ∂²f / ∂x_i ∂x_j`  
Symmetric (for smooth f).

**Analogy:** Curvature in every pair of directions. Positive definite → local minimum, negative definite → maximum, indefinite → saddle.

### 3.2 Hessian in Optimization
- Newton's method: `Δx = -H⁻¹ ∇f` – uses curvature to accelerate.
- Second-order optimization in ML (L-BFGS) approximates Hessian.
- **Hessian-free optimization** for RNNs.

### 3.3 Connection to Neural Network Training
The **Fisher Information Matrix** (for probabilistic models) equals expected Hessian of negative log-likelihood. Used in natural gradient descent.

```python
# Approximate Hessian via finite differences (for small n)
def hessian(f, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Central difference formula
            ...
    return H
```

### 📝 Quiz
What does a zero Hessian eigenvalue imply? (Flat direction – no curvature along that eigenvector.)

---

## Module 4: Divergence & Curl – Flows and Rotations <a name="divcurl"></a>

### 4.1 Divergence (for vector fields)
`div F = ∇·F = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z`  
Measures net outflow from a point.  
**Analogy:** Expansion/compression of fluid. Positive divergence = source, negative = sink.

**In ML:**
- **Normalizing flows:** Learn invertible transformations with divergence to compute log-density change.
- **Score-based models:** The score `∇ log p(x)` has divergence = Laplacian of log density.

### 4.2 Curl (3D only)
`curl F = ∇ × F` – measures rotation around a point.  
**Analogy:** Vorticity in a whirlpool.

### 4.3 Laplacian (Divergence of Gradient)
`Δf = ∇·∇f = Σ ∂²f/∂x_i²` – measures "spread" of function.  
**In AI:**  
- **Heat equation** for smoothing.  
- **Graph Laplacian** for spectral clustering.  
- **Diffusion models:** Reverse SDE involves Laplacian of score.

```python
# Laplacian of 2D Gaussian
def laplacian(f, x, y, h=0.01):
    # finite difference approximation
    f_xy = f(x,y)
    f_xp = f(x+h, y); f_xm = f(x-h, y)
    f_yp = f(x, y+h); f_ym = f(x, y-h)
    return (f_xp+f_xm+f_yp+f_ym - 4*f_xy)/h**2
```

### 📝 Exercise
What is divergence of gradient field of `f(x,y)=x²+y²`? (Laplacian = 4, constant)

---

## Module 5: Line Integrals – Accumulation Along Paths <a name="line-integrals"></a>

### 5.1 Definition
`∫_C F·dr` = work done by vector field F along path C.  
For gradient fields: `∫_C ∇f·dr = f(end) - f(start)` (path independent).

**Analogy:** Work = force dot displacement. If force is conservative (gradient), work depends only on endpoints.

### 5.2 In Policy Gradients (Reinforcement Learning)
The expected return objective: `J(θ) = E[ Σ r_t ]`.  
The policy gradient theorem shows: `∇J = E[ Σ ∇ log π(a_t|s_t) * Q(s_t,a_t) ]` – a kind of line integral in policy space.

### 5.3 In Generative Models
- **Continuous normalizing flows (CNFs):** log-density change = integral of divergence along flow.
- **Neural ODEs:** `dz/dt = f(z,t)`, loss depends on entire trajectory – need adjoint method (reverse-time integration).

```python
# Simple line integral of F(x,y)=[y, -x] along unit circle
import numpy as np
theta = np.linspace(0, 2*np.pi, 1000)
x = np.cos(theta); y = np.sin(theta)
dx = -np.sin(theta); dy = np.cos(theta)
F_dot_dr = y*dx + (-x)*dy  # = -1 dt
work = np.trapz(F_dot_dr, theta)  # = -2π
```

---

## Module 6: Backpropagation as Vector Chain Rule <a name="backprop-vector"></a>

### 6.1 Chain Rule for Vector Functions
If `h = g(f(x))`, then `J_h = J_g * J_f` (matrix multiplication).

### 6.2 Backprop Algorithm
Given loss `L = L(y)`, with `y = f_n(... f_1(x))`:
- Forward: compute all intermediate values.
- Backward: `∂L/∂x = J_1ᵀ * J_2ᵀ * ... * J_nᵀ * (∂L/∂y)`  
  (Reverse order, transpose of Jacobians).

**Analogy:** Errors flow backward through a pipeline, each stage transforms the error vector by the transpose Jacobian.

### 6.3 Example: Two Linear Layers with ReLU
```
x → z1 = W1 x → a1 = ReLU(z1) → z2 = W2 a1 → loss
```
Backprop:
```
∂L/∂z2 = ∂L/∂a2 (since a2 = z2 for linear output)
∂L/∂W2 = (∂L/∂z2) a1ᵀ
∂L/∂a1 = W2ᵀ (∂L/∂z2)
∂L/∂z1 = ∂L/∂a1 * diag(1_{z1>0})   # ReLU Jacobian
∂L/∂W1 = (∂L/∂z1) xᵀ
```

### 6.4 Memory Efficiency (Gradient Checkpointing)
Store only few activations, recompute others during backward – trades compute for memory.

### 📝 Exercise
Why do we need the transpose of Jacobian in backprop? (Because we propagate derivatives with respect to inputs, and J is defined as ∂output/∂input – chain rule gives product of Jacobians, but we multiply by gradient row vector from the right, effectively using Jᵀ.)

---

## Module 7: Applications in Generative AI: Score Functions & Diffusion <a name="generative-vector"></a>

### 7.1 Score Function
`∇_x log p(x)` – gradient of log density with respect to data.  
**Properties:**
- Points toward high density.
- Zero at modes.
- Divergence = negative Fisher information.

### 7.2 Denoising Score Matching
Learn score by denoising: `E[‖s_θ(x̃) - ∇_x log p(x̃|x)‖²]` where x̃ = noisy version.

### 7.3 Diffusion Models (DDPM + Score-SDE)
Forward SDE: `dx = f(x,t) dt + g(t) dw`  
Reverse SDE: `dx = [f(x,t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw̄`  
The score `∇_x log p_t(x)` is learned by a neural network.

### 7.4 Probability Flow ODE
Convert SDE to ODE: `dx/dt = f(x,t) - ½ g(t)² ∇_x log p_t(x)` – deterministic trajectory that matches marginal distributions.

### 7.5 Vector Calculus Connection
- Score = gradient field.
- Divergence of score = Laplacian of log density (appears in Fokker-Planck equation).
- Line integral of score along a path = log-density difference.

```python
# Simplified diffusion training (score matching)
def diffusion_loss(model, x):
    t = np.random.uniform(0,1)
    noise = np.random.randn(*x.shape)
    x_t = sqrt(1-β_t)*x + sqrt(β_t)*noise
    pred_score = model(x_t, t)
    target_score = -noise / sqrt(β_t)  # for VP SDE
    return np.mean((pred_score - target_score)**2)
```

---

## Module 8: Applications in Agentic AI: Policy Gradients & Q-Learning <a name="agentic-vector"></a>

### 8.1 Policy Gradient Theorem (Reinforce)
For stochastic policy `π_θ(a|s)`, the gradient of expected return is:  
`∇J(θ) = E[ Σ ∇_θ log π_θ(a_t|s_t) * Q^π(s_t, a_t) ]`

The term `∇_θ log π` is the gradient of the log-likelihood of action – a vector with respect to θ.

### 8.2 Natural Policy Gradient
Uses Fisher Information Matrix (expected outer product of score) to precondition gradient:  
`∇_natural = F(θ)⁻¹ ∇J`, where `F(θ) = E[∇ log π (∇ log π)ᵀ]` – a positive semidefinite matrix.

### 8.3 Deterministic Policy Gradients (DPG)
For deterministic policy `μ_θ(s)`, gradient:  
`∇J(θ) = E[ ∇_θ μ_θ(s) * ∇_a Q(s,a)|_{a=μ_θ(s)} ]` – product of Jacobian of μ and gradient of Q.

### 8.4 Actor-Critic Methods
- **Actor:** Policy `π_θ`, gradient as above.
- **Critic:** Value function `V_φ` or `Q_φ`, trained with TD error.
- The critic gradient is a standard regression gradient.

### 8.5 Hessian-Free Optimization in TRPO
Trust Region Policy Optimization constrains step size using KL divergence, which locally approximates Fisher information. Requires solving `H p = g` (conjugate gradient with Hessian-vector products).

```python
# Policy gradient (REINFORCE) with baseline
def reinforce(env, policy, optimizer, gamma=0.99):
    states, actions, rewards = rollout(env, policy)
    returns = discounted_cumsum(rewards, gamma)
    # standardize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    log_probs = policy.log_prob(states, actions)
    loss = - (log_probs * returns).mean()
    optimizer.zero_grad()
    loss.backward()  # computes gradient of log_probs * returns
    optimizer.step()
```

---

## Module 9: Matrix Calculus – Derivatives with Respect to Matrices <a name="matrix-calculus"></a>

### 9.1 Motivation
Neural network weights are matrices (e.g., `W` in linear layer). We need derivatives like `∂L/∂W`.

### 9.2 Matrix Derivatives – Notation
We can flatten matrices to vectors, but better to use matrix calculus with **denominator layout**.

| Type | Shape of ∂y/∂x | Example |
|------|----------------|---------|
| scalar / scalar | 1 | |
| scalar / matrix | same shape as matrix | ∂L/∂W (matrix) |
| vector / scalar | vector | |
| matrix / scalar | matrix | |

### 9.3 Useful Matrix Derivatives

| Expression | Derivative |
|------------|-------------|
| `∂(aᵀx)/∂x` | aᵀ (row) |
| `∂(xᵀAx)/∂x` | 2xᵀA (if A sym) |
| `∂(Wx)/∂W` | Jacobian: plays role of xᵀ ⊗ I (Kronecker) |
| `∂L/∂W` for `z = Wx`, `L = loss(z)` | `∂L/∂W = (∂L/∂z) xᵀ` |

### 9.4 Kronecker Product & Vec Operator
`vec(Wx) = (xᵀ ⊗ I) vec(W)` – helps compute matrix gradients systematically.

### 9.5 Backprop Through Fully Connected Layer (Matrix Form)
```
Forward: Z = X W + b
Backward:
dL/dW = Xᵀ (dL/dZ)
dL/dX = (dL/dZ) Wᵀ
dL/db = sum(dL/dZ, axis=0)
```

```python
# NumPy implementation
def linear_backward(dZ, X, W):
    dW = X.T @ dZ
    dX = dZ @ W.T
    db = np.sum(dZ, axis=0)
    return dX, dW, db
```

### 📝 Exercise
Derive `∂L/∂W` for `L = ½‖Y - XW‖²_F` (Frobenius norm).  
Answer: `Xᵀ (XW - Y)`

---

## Module 10: Final Project – Implement Automatic Differentiation <a name="final-project-vector"></a>

### Build a Mini Autograd Engine

**Requirements:**
1. Implement `Value` class with scalar values and computational graph.
2. Support operations: `+`, `*`, `/`, `exp`, `log`, `relu`, `sigmoid`.
3. Implement `backward()` using topological sort (reverse mode autodiff).
4. Add a `Linear` layer using matrix version.
5. Train a two-layer neural network on XOR.

**Sample Code Structure:**
```python
class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

**What you'll learn:**
- How automatic differentiation implements vector chain rule.
- The relationship between Jacobians and gradients.
- Why backpropagation stores intermediate values.

---

## Next Steps & Resources

### Books
- **"Calculus"** by Stewart (intuitive)
- **"Matrix Calculus"** by Magnus & Neudecker (rigorous)
- **"Deep Learning"** by Goodfellow et al. (Chapter 4 on calculus)

### Online Courses
- **Khan Academy Calculus** (visual)
- **MIT 18.02 Multivariable Calculus** (classic)
- **CS224n: Natural Language Processing with Deep Learning** (vector calculus for backprop)

### Libraries to Master for Calculus
- **PyTorch/TensorFlow** (autograd is vector calculus in action)
- **JAX** (functional autograd with grad, jacobian, hessian)
- **SymPy** (symbolic differentiation for verification)

---

## Memory Palace Summary (Vector Calculus)

| Concept | Memory Aid |
|---------|-------------|
| Jacobian | "Just All Combinations" – matrix of partials |
| Hessian | "How Each Second derivative... All directions" |
| Divergence | "Dispersal" – net outflow |
| Curl | "Circular rotation" |
| Laplacian | "Lazy uniformation" – smoothing |
| Score function | Gradient of log probability |
| Fisher info | Expected square of score |
| Natural gradient | Preconditioned gradient |
| Autograd | Automatic chain rule |

---

## Appendix: Quick Reference – Derivatives in ML

### Common Jacobians

| Function f(x) | Jacobian ∂f/∂x |
|---------------|----------------|
| `σ(x)` (sigmoid, elementwise) | `diag(σ(x) ⊙ (1-σ(x)))` |
| `tanh(x)` | `diag(1-tanh²(x))` |
| `softmax(x)` | `diag(s) - s sᵀ` where s = softmax(x) |
| `Wx` | `W` |
| `ReLU(x)` | `diag(1_{x>0})` |

### Useful Identities

- `∂/∂x (xᵀ A x) = (A + Aᵀ)x`
- `∂/∂W (‖Y - XW‖²) = -2 Xᵀ (Y - XW)`
- `∂L/∂x = (∂z/∂x)ᵀ (∂L/∂z)` for z = f(x)
- `vec(AXB) = (Bᵀ ⊗ A) vec(X)`

---

**Congratulations!** You've completed the three-part calculus series for AI.

From gradients to Jacobians to diffusion scores, you now understand the mathematical engine that powers modern machine learning, generative AI, and intelligent agents.

**Your final checklist:**
- [ ] I can compute derivatives and use chain rule.
- [ ] I understand gradients and gradient descent.
- [ ] I can compute partial derivatives and read Jacobians.
- [ ] I know how backprop uses vector chain rule.
- [ ] I can explain score functions and diffusion models.
- [ ] I grasp policy gradients and natural gradients.

Now you're ready to read research papers, implement a
