# Calculus for ML/DL/Generative AI: Zero to Hero (Part 2)

## Part 2: Multivariable Calculus – Navigating High Dimensions

**File:** `calculus2_multivariable.md`

---

# Calculus II: Multivariable Calculus for Machine Learning

**From Layman to Data Scientist**  
*Analogies • Memory Techniques • Hands-On Examples*

---

## Table of Contents

1. [Why Multivariable?](#why-multivariable)
2. [Functions of Several Variables](#multivariable-functions)
3. [Partial Derivatives – Focus on One Axis](#partial-derivatives)
4. [The Gradient – Vector of Steepest Ascent](#gradient)
5. [Directional Derivatives – Any Direction](#directional)
6. [Optimization in Multiple Dimensions](#multivariate-optimization)
7. [Gradient Descent in High Dimensions](#gd-highdim)
8. [Lagrange Multipliers – Constraints](#lagrange)
9. [Double & Triple Integrals](#multiple-integrals)
10. [Applications in DL & Generative Models](#applications-multivariable)

---

## Module 1: Why Multivariable? <a name="why-multivariable"></a>

**Analogy:** Single variable is a path along a ridge. Multivariable is the entire mountain range – you need to know slope in every direction.

ML models have thousands to billions of parameters (weights). Optimizing over such high-dimensional spaces requires multivariable calculus.

**Memory Trick:** **Multivariable = "Many Variables"** – each partial derivative tells sensitivity to one weight.

---

## Module 2: Functions of Several Variables <a name="multivariable-functions"></a>

### 2.1 Definition
A function `f(x₁, x₂, ..., xₙ)` maps multiple inputs to a scalar (e.g., loss) or vector (e.g., layer output).

**Analogy:** Your happiness (output) depends on many factors: coffee intake, sleep hours, work stress.

### 2.2 Visualizing in 3D
For two variables, `z = f(x,y)` creates a surface.  
- **Contour plots:** Level sets where f is constant.  
**Analogy:** Topographic map of a mountain – each contour is constant altitude.

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
X,Y = np.meshgrid(x,y)
Z = X**2 + Y**2   # paraboloid
plt.contourf(X,Y,Z)
plt.colorbar()
```

---

## Module 3: Partial Derivatives – Focus on One Axis <a name="partial-derivatives"></a>

### 3.1 Definition
`∂f/∂x` = derivative with respect to x, treating other variables as constants.

**Analogy:** How your grade changes if you study one more hour for math, holding physics constant.

### 3.2 Computation
`f(x,y) = 3x²y + sin(y)`  
`∂f/∂x = 6xy`  
`∂f/∂y = 3x² + cos(y)`

**Memory Trick:** **Partial = "Part"** – change only one part at a time.

### 3.3 Higher-Order Partials
`∂²f/∂x²` (pure second), `∂²f/∂x∂y` (mixed partial).  
For smooth functions, mixed partials are equal (Clairaut's theorem): `∂²f/∂x∂y = ∂²f/∂y∂x`

### 📝 Exercise
Compute `∂f/∂x` and `∂f/∂y` for `f(x,y)=x·e^(xy)`.  
(Answer: ∂f/∂x = e^(xy)(1+xy), ∂f/∂y = x² e^(xy))

---

## Module 4: The Gradient – Vector of Steepest Ascent <a name="gradient"></a>

### 4.1 Definition
The gradient is a vector of all partial derivatives:  
`∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`

**Properties:**
- Points in direction of fastest increase of f.
- Magnitude equals maximum rate of change.
- Perpendicular to level sets (contours).

**Analogy:** A hiker feeling the steepest uphill direction – that's the gradient. For downhill, take negative gradient.

**Memory Trick:** **Gradient = "Go Rampant In Direction of Extreme Tallness"** (GRIDET – but just remember: gradient points uphill).

### 4.2 Gradient in ML
The loss function `L(w)` – we compute `∇L` to take a step opposite:  
`w ← w - α ∇L(w)`

### 4.3 Gradient of Common ML Functions

| Function | Gradient (w.r.t w) |
|----------|---------------------|
| Linear model `L = ½‖Xw - y‖²` | `Xᵀ(Xw-y)` |
| Logistic loss | `Xᵀ(σ(Xw) - y)` |
| Neural network layer | Chain rule through activation |

```python
# Gradient of MSE loss for linear regression
def mse_gradient(w, X, y):
    return X.T @ (X @ w - y) / len(X)
```

### 📝 Quiz
What is the gradient of `f(x,y)=x²+y²` at (1,2)? (Answer: (2,4))

---

## Module 5: Directional Derivatives – Any Direction <a name="directional"></a>

### 5.1 Definition
Derivative in direction of unit vector `u`: `D_u f = ∇f · u`

**Analogy:** You want to know how steep the hill is if you walk northeast, not just north or east.

### 5.2 Maximum & Minimum
- Max: direction of `∇f` (magnitude = ‖∇f‖)
- Min: direction of `-∇f` (magnitude = -‖∇f‖)
- Zero: directions tangent to level sets.

### 5.3 In ML Training
Stochastic gradient descent actually uses noisy estimates of the directional derivative along batch direction.

---

## Module 6: Optimization in Multiple Dimensions <a name="multivariate-optimization"></a>

### 6.1 Critical Points
`∇f = 0` (all partials zero). Types:
- **Local minimum:** `∇²f` positive definite (Hessian has all positive eigenvalues)
- **Local maximum:** `∇²f` negative definite
- **Saddle point:** mixed eigenvalues (common in high-D deep learning)

**Analogy:** Saddle point = mountain pass – goes up in one direction, down in another.

### 6.2 The Hessian Matrix (Preview of Vector Calculus)
Matrix of second partials: `H_ij = ∂²f/∂x_i∂x_j`.  
Eigenvalues tell curvature.

### 6.3 Newton's Method (Second-order optimization)
`w_new = w_old - H⁻¹ ∇f` – converges faster but expensive.

---

## Module 7: Gradient Descent in High Dimensions <a name="gd-highdim"></a>

### 7.1 Batch vs. Stochastic vs. Mini-batch
- **Batch GD:** Use all data to compute `∇L`.
- **Stochastic GD (SGD):** Use one random sample.
- **Mini-batch:** Use a small batch (compromise).

**Analogy:** Finding a lost key. Batch = scan whole room (accurate but slow). SGD = guess randomly (fast but noisy). Mini-batch = scan a corner (balanced).

### 7.2 Challenges
- **Ill-conditioning:** Hessian has very different eigenvalues → slow convergence.
- **Local minima & saddle points:** In high dimensions, saddles are more common than local minima.
- **Vanishing/exploding gradients:** Chain rule product too small/large.

### 7.3 Momentum & Nesterov
`v ← β v + α ∇L`  
`w ← w - v`  
**Analogy:** Rolling ball with inertia – smooths out oscillations.

### 7.4 Adaptive Methods (Adam, RMSprop)
Maintain per-parameter learning rates using moving averages of squared gradients.

```python
# Simple SGD with momentum
v = 0
beta = 0.9
alpha = 0.01
for epoch in range(epochs):
    grad = compute_gradient(w)
    v = beta * v + alpha * grad
    w = w - v
```

### 📝 Exercise
Why does SGD with momentum help escape shallow local minima? (Inertia carries it through small bumps)

---

## Module 8: Lagrange Multipliers – Constraints <a name="lagrange"></a>

### 8.1 Problem
Optimize `f(x)` subject to constraint `g(x)=0`.

### 8.2 Solution
Introduce Lagrange multiplier λ:  
`∇f = λ ∇g` simultaneously with `g(x)=0`.

**Analogy:** A fence (constraint) on a hill. The highest point on the fence is where the fence's direction is tangent to the hill's contours – i.e., gradients are parallel.

### 8.3 ML Applications
- **Regularization:** L2 penalty = constraint on weight norm.
- **SVM:** Maximize margin subject to classification constraints.
- **Normalization layers:** Constrain activations to zero mean and unit variance.

### 8.4 Example: Norm Constraint
Minimize `f(w)` subject to `‖w‖² = c`. Lagrangian: `L(w,λ)=f(w) + λ(‖w‖²-c)`.

### 📝 Quiz
Interpretation of λ at optimum? (Sensitivity of optimal value to constraint tightness)

---

## Module 9: Double & Triple Integrals <a name="multiple-integrals"></a>

### 9.1 Double Integral Over Region
`∬_R f(x,y) dA` = volume under surface.

**Analogy:** Total rainfall over a map – integrate rain intensity over area.

### 9.2 In Probability
Joint probability density: `∬ p(x,y) dx dy = 1`.  
Marginal density: `p(x) = ∫ p(x,y) dy`.

### 9.3 Applications in Generative Models
- **Latent variable models:** `p(x) = ∫ p(x|z) p(z) dz` – integral over latent space.
- **VAEs:** Approximate this intractable integral with variational lower bound.

### 9.4 Numerical Integration in High Dimensions
Monte Carlo methods: sample from distribution, average function values.  
**Why:** Curse of dimensionality makes grid methods infeasible.

```python
# Monte Carlo estimate of ∫∫ (x²+y²) over unit square
samples = np.random.rand(100000, 2)
f = samples[:,0]**2 + samples[:,1]**2
estimate = np.mean(f)   # ≈ 0.667
```

---

## Module 10: Applications in DL & Generative Models <a name="applications-multivariable"></a>

### 10.1 Training Neural Networks
- **Loss landscape** is a high-dimensional function: `L(W1, W2, ..., b1, b2, ...)`
- Gradient descent on all parameters simultaneously.

### 10.2 Batch Normalization
Normalizes activations per batch: `(x - μ)/σ` – involves gradients through μ and σ (requires careful multivariable chain rule).

### 10.3 Variational Autoencoders (VAEs)
Optimize ELBO: `E_q[log p(x|z)] - KL(q(z|x) || p(z))`.  
The KL term is an integral over q. Reparameterization trick reparameterizes z = μ + σ·ε to push gradient through.

### 10.4 Generative Adversarial Networks (GANs)
The generator's output distribution implicitly defines a high-dimensional probability. No explicit density – but gradient flows through the discriminator.

### 10.5 Diffusion Models
Forward process: `x_t = √(1-β_t) x_{t-1} + √β_t ε_t` – a stochastic differential equation. Reverse process learns score function `∇ log p_t(x)` – a gradient in data space.

### 📝 Code Snippet: VAE Reparameterization
```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z   # gradient flows through mu and std
```

---

## Next: Vector Calculus
You've mastered partial derivatives and gradients. Now we'll tackle vector-valued functions, Jacobians, and the math behind backpropagation and attention.

---

**Memory Palace Summary (Multivariable)**

| Concept | Memory Aid |
|---------|-------------|
| Partial derivative | Change one thing at a time |
| Gradient | Steepest uphill vector |
| Directional derivative | Gradient dot direction |
| Saddle point | Up one way, down another |
| Lagrange multiplier | Gradients parallel at constraint boundary |
| Reparameterization trick | Push gradient through random node |

---

