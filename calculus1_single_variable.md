# Calculus for ML/DL/Generative AI: Zero to Hero (3-Part Series)

## Part 1: Single Variable Calculus – The Foundation of Change

**File:** `calculus1_single_variable.md`

---

# Calculus I: Single Variable Calculus for Machine Learning

**From Layman to Data Scientist**  
*Analogies • Memory Techniques • Hands-On Examples*

---

## Table of Contents

1. [Why Calculus for AI?](#why-calculus)
2. [Limits – Approaching Infinity](#limits)
3. [Derivatives – The Instantaneous Rate of Change](#derivatives)
4. [Differentiation Rules – Shortcuts](#rules)
5. [Chain Rule – The Heart of Backprop](#chain-rule)
6. [Optimization – Finding Minima](#optimization)
7. [Integration – Accumulation](#integration)
8. [Applications in ML: Gradient Descent, Learning Rate, etc.](#applications)
9. [Exercises & Next Steps](#exercises)

---

## Module 1: Why Calculus for AI? <a name="why-calculus"></a>

**Analogy:** If machine learning is a car, calculus is the engine. It tells you how to change the car's parameters to reach your destination (minimum loss) faster.

**Memory Trick:** **Calculus = "Calculate How Change Accumulates"**

### Where Calculus Appears in AI
- **Training neural networks:** Derivatives tell us how to adjust weights.
- **Optimization algorithms:** Gradient descent, Adam, etc.
- **Loss functions:** Cross-entropy, MSE derivatives.
- **Generative models:** Diffusion models use score functions (derivatives of log probability).
- **Agentic AI:** Policy gradients use derivatives to improve rewards.

---

## Module 2: Limits – Approaching Infinity <a name="limits"></a>

### 2.1 What is a Limit?
The value a function approaches as input approaches some point.  
`lim_{x→a} f(x) = L`

**Analogy:** You walk toward a wall. As you get closer, your distance approaches zero. The limit is zero, even if you never touch it.

### 2.2 Why Limits Matter in ML
- **Derivatives** are defined using limits.
- **Gradient clipping** uses limits to prevent exploding gradients.
- **Convergence proofs** of optimization algorithms rely on limits.

### 2.3 One-Sided Limits & Continuity
A function is continuous if the limit equals the function value.  
**Analogy:** A smooth path vs. a jump (discontinuity). Neural networks need smooth activations like sigmoid/tanh (ReLU is not differentiable at 0 but works in practice).

```python
# Visualizing limit of 1/x as x → ∞
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0.1, 10, 100)
y = 1/x
plt.plot(x, y)
plt.title("As x → ∞, 1/x → 0")
```

### 📝 Quick Quiz
What is `lim_{x→0} sin(x)/x`? (Answer: 1 – fundamental for derivative of sin)

---

## Module 3: Derivatives – The Instantaneous Rate of Change <a name="derivatives"></a>

### 3.1 Definition & Intuition
`f'(x) = lim_{h→0} [f(x+h) - f(x)] / h`

**Analogy:** Speedometer reading at an instant. Average speed = total distance/time. Instantaneous speed = derivative of position.

**Memory Trick:** **Derivative = "Divide by h, let h shrink"** – **D**erivative **i**s **s**lope at a **p**oint (DISP).

### 3.2 Geometric Meaning
Slope of tangent line. Positive derivative = function increasing. Negative = decreasing. Zero = flat (possible extremum).

### 3.3 Common Derivatives (Memory Palace)

| Function | Derivative | Memory Aid |
|----------|------------|-------------|
| Constant c | 0 | Flat line → no change |
| xⁿ | n·xⁿ⁻¹ | "Power down" |
| eˣ | eˣ | "Exponential stays itself" |
| ln(x) | 1/x | "ln drops to reciprocal" |
| sin(x) | cos(x) | "sine to cosine" |
| cos(x) | -sin(x) | "cosine negative sine" |
| σ(x) = 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) | "Sigmoid times its complement" |
| tanh(x) | 1 - tanh²(x) | "Tanh keeps itself squared away" |

### 3.4 Derivatives in ML – The Loss Landscape
Loss function `L(w)` where w is a model parameter. Derivative `dL/dw` tells how loss changes if we tweak w.

```python
# Derivative of MSE loss: L = (y - ŷ)², ŷ = w*x
def mse_derivative(w, x, y):
    y_pred = w * x
    return -2 * x * (y - y_pred)
```

### 📝 Exercise
Derivative of `f(x) = 3x² + 2x + 1` at x=2? (Answer: 6*2+2=14)

---

## Module 4: Differentiation Rules – Shortcuts <a name="rules"></a>

### 4.1 Basic Rules
- **Sum Rule:** (f+g)' = f' + g'
- **Product Rule:** (fg)' = f'g + fg'  
  **Memory:** "First derivative of second plus second derivative of first"
- **Quotient Rule:** (f/g)' = (f'g - fg')/g²  
  **Memory:** "Low d-high minus high d-low, over low squared"

### 4.2 Why Not Always Definition?
We chain hundreds of operations in neural nets – rules make it efficient.

### 4.3 The Chain Rule (Most Important for AI)
`(f(g(x)))' = f'(g(x)) * g'(x)`  
**Analogy:** If you drive twice as fast (g) and your fuel consumption per km doubles (f), total consumption multiplies.

**Memory Trick:** **Chain = "Chug along"** – differentiate outer, then inner, multiply.

```python
# Example: f(x) = (3x²+1)⁴
# Outer: u⁴ → 4u³, Inner: 3x²+1 → 6x
# Derivative = 4(3x²+1)³ * 6x = 24x(3x²+1)³
```

---

## Module 5: Chain Rule – The Heart of Backprop <a name="chain-rule"></a>

### 5.1 Backpropagation is Chain Rule on Steroids
A neural network is a composition of functions:  
`Loss = L( f_k( ... f_2( f_1(x, W1), W2) ..., Wk)`

The derivative of loss w.r.t. any weight is a product of derivatives through all subsequent layers.

**Analogy:** A telegraph message passing through operators. Each operator modifies the message (derivative), and the final effect is the product of all modifications.

### 5.2 Simple Example: Two Linear Layers
```
x → h = W1·x → y = W2·h → loss = (y - target)²
∂loss/∂W2 = 2(y-target) * hᵀ
∂loss/∂W1 = 2(y-target) * W2ᵀ * xᵀ
```

### 5.3 Computational Graph Visualization
```
x → (W1) → h → (W2) → y → loss
```
Arrows represent chain rule multiplication.

### 📝 Exercise
If `f(x)=sin(x²)`, compute f'(x). (Chain rule: cos(x²) * 2x)

---

## Module 6: Optimization – Finding Minima <a name="optimization"></a>

### 6.1 Critical Points
Points where `f'(x)=0`: local minima, maxima, or saddle points.

**Analogy:** Mountain valleys (minima), peaks (maxima), flat passes (saddles).

### 6.2 First Derivative Test
If derivative changes from negative to positive → local minimum.

### 6.3 Second Derivative Test
`f''(x) > 0` → convex (minimum), `f''(x) < 0` → concave (maximum).
**Analogy:** Acceleration tells you if you're at the bottom of a valley (curving up) or top of a hill (curving down).

### 6.4 Gradient Descent (for single variable)
`w_new = w_old - α * f'(w_old)` where α = learning rate.

**Analogy:** Blindfolded person going downhill by feeling the slope under their feet. Too big step (α large) might overshoot, too small → slow.

```python
# Gradient descent for f(w)=w²
w = 5.0
alpha = 0.1
for step in range(20):
    grad = 2 * w
    w = w - alpha * grad
    print(f"Step {step}: w={w:.4f}")
```

### 6.5 Learning Rate Tuning
- Too high: overshoot, diverge.
- Too low: slow convergence.
- Adaptive methods (Adam) use per-parameter learning rates.

### 📝 Quiz
If `f'(x)=0` and `f''(x)=0`, what could the point be? (Saddle or inflection – flat region)

---

## Module 7: Integration – Accumulation <a name="integration"></a>

### 7.1 Definite Integral as Area Under Curve
`∫_a^b f(x) dx` = total accumulated value.

**Analogy:** Distance traveled = integral of velocity over time. Revenue = integral of sales rate.

### 7.2 Fundamental Theorem of Calculus
Derivative and integral are inverses:  
`d/dx ∫_a^x f(t) dt = f(x)`

### 7.3 Integration in ML
- **Expected value** in probability: `E[X] = ∫ x p(x) dx`
- **Bayesian inference:** Marginal likelihood is integral over parameters.
- **Reinforcement learning:** Value functions are expectations (integrals over rewards).
- **Diffusion models:** Forward process integrates noise over time.

### 7.4 Simple Numerical Integration (Monte Carlo)
```python
# Approximate ∫₀¹ x² dx
import numpy as np
samples = np.random.uniform(0,1,10000)
estimate = np.mean(samples**2)  # ≈ 1/3
```

### 📝 Exercise
Why is integration important for continuous probability distributions? (Area under PDF must be 1; probabilities are integrals of PDF)

---

## Module 8: Applications in ML <a name="applications"></a>

### 8.1 Activation Function Derivatives

| Activation | f(x) | f'(x) | Use Case |
|------------|------|-------|----------|
| Sigmoid | 1/(1+e⁻ˣ) | f(x)(1-f(x)) | Output probability |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1 - f(x)² | Hidden layer (zero-centered) |
| ReLU | max(0,x) | 0 if x<0 else 1 | Deep networks (sparse) |
| Softplus | ln(1+eˣ) | sigmoid(x) | Smooth ReLU |

### 8.2 Loss Function Derivatives

- **MSE:** `L = ½(y-ŷ)²` → `∂L/∂ŷ = -(y-ŷ)`
- **Binary Cross-Entropy:** `L = -[y log ŷ + (1-y) log(1-ŷ)]` → `∂L/∂ŷ = -(y/ŷ) + (1-y)/(1-ŷ)`
- **Categorical Cross-Entropy** (softmax): derivative simplifies to `ŷ - y` (after softmax)

### 8.3 Gradient Descent Variants
- **Momentum:** uses exponential moving average of past gradients.
- **Adam:** combines momentum with adaptive learning rates.

### 📝 Code Example: Training a Single Neuron
```python
import numpy as np

# Sigmoid neuron with MSE loss
def train(X, y, epochs=100, lr=0.1):
    w, b = np.random.randn(X.shape[1]), 0.0
    for epoch in range(epochs):
        z = X @ w + b
        y_pred = 1/(1+np.exp(-z))
        loss = np.mean((y - y_pred)**2)
        # Derivatives
        dl_dy_pred = -2*(y - y_pred)
        dy_pred_dz = y_pred*(1-y_pred)
        dl_dz = dl_dy_pred * dy_pred_dz
        grad_w = X.T @ dl_dz / len(X)
        grad_b = np.mean(dl_dz)
        # Update
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
```

---

## Module 9: Exercises & Next Steps <a name="exercises"></a>

### Hands-on Problems
1. Compute derivative of `f(x)=ln(1+eˣ)` (softplus). Answer: sigmoid(x)
2. Derive gradient of MSE loss with respect to prediction.
3. Implement gradient descent for `f(x)=x⁴ - 3x² + 2x` and find minimum.
4. Explain why ReLU derivative is 0 for negative inputs (dead neurons).

### Next: Multivariable Calculus
You've mastered single variable. Now we'll explore gradients in higher dimensions – the true workhorse of deep learning.

---

**Memory Palace Summary (Single Variable)**

| Concept | Memory Aid |
|---------|-------------|
| Limit | Approach, not touch |
| Derivative | Divide by h, let h shrink (DISP) |
| Product rule | First d second + second d first |
| Chain rule | Chug along (outer' * inner') |
| Gradient descent | Step opposite slope |
| Integration | Area under curve |

---

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

Now you're ready to read research papers, implement algorithms from scratch, and push the boundaries of AI. Go create something amazing! 🚀
