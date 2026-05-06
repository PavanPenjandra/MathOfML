# 📐 Mathematics for Machine Learning, Deep Learning & Generative AI – Zero to Hero

**From Layman to Data Scientist**  
*A complete 5-course series covering everything you need to master the math behind modern AI.*

---

## 🎯 Why This Series?

Every ML/DL/GenAI algorithm is built on three mathematical pillars:

- **Linear Algebra** – the language of data & transformations  
- **Statistics & Probability** – the science of uncertainty & inference  
- **Calculus (Single, Multi, Vector)** – the engine of optimization & learning  

This series transforms you from zero to confident practitioner – no hidden prerequisites, no hand‑waving. Each concept uses **analogies, memory techniques, and executable Python examples** so you truly understand, not just memorize.

> **What makes this series unique?**  
> - **Zero‑to‑hero** – begins with scalars, ends with automatic differentiation & diffusion scores.  
> - **AI‑centric** – every topic is directly tied to ML, DL, generative AI, or agentic AI.  
> - **Memory palaces** – forget‑proof mnemonics for every major idea.  
> - **Code first** – NumPy/PyTorch snippets show the math in action.

---

## 📚 Course Roadmap

| # | Course | Focus | Key Outcomes |
|---|--------|-------|---------------|
| 1 | [Linear Algebra](#1-linear-algebra) | Vectors, matrices, eigendecomposition, SVD | PCA, word embeddings, attention, latent spaces |
| 2 | [Statistics & Probability](#2-statistics--probability) | Descriptive stats, inference, MLE, Bayesian, hypothesis testing | Model evaluation, uncertainty, A/B testing, generative metrics |
| 3 | [Calculus I – Single Variable](#3-calculus-i--single-variable) | Derivatives, chain rule, optimization, integration | Gradient descent, backprop foundations, learning rates |
| 4 | [Calculus II – Multivariable](#4-calculus-ii--multivariable) | Partial derivatives, gradient, Lagrange multipliers | Training neural nets, VAE reparameterization, constraints |
| 5 | [Calculus III – Vector Calculus](#5-calculus-iii--vector-calculus) | Jacobians, Hessians, divergence, policy gradients | Backpropagation, diffusion score functions, natural gradients |

---

## 1. Linear Algebra

**File:** [`linear_algebra_for_ml.md`](linear_algebra_for_ml.md)

### What you'll learn
- Scalars, vectors, matrices – and why they are spreadsheets + transformations  
- Matrix multiplication (the “inner dimensions must match” rule)  
- Eigenvalues & eigenvectors – the heart of PCA and PageRank  
- Singular Value Decomposition (SVD) – the Swiss army knife of data science  
- Norms, distances, and cosine similarity for embeddings  
- Applications: word2vec, attention (`QKᵀ`), latent space navigation in GANs/VAEs

### Memory gems
- **Dot product** = “Do the sum of products”  
- **Determinant** = “Does Everything Twist?” (zero = flat)  
- **SVD** = “Superb Valuable Decomposition”

---

## 2. Statistics & Probability

**File:** [`statistics_for_ml.md`](statistics_for_ml.md)

### What you'll learn
- Descriptive statistics: mean/median/mode, variance, skewness  
- Probability rules, conditional probability, Bayes’ theorem  
- Discrete & continuous distributions (Binomial, Normal, Poisson)  
- Central Limit Theorem – why averages become normal  
- Hypothesis testing, p‑values, confidence intervals  
- Maximum Likelihood Estimation (MLE) – link to loss functions  
- Bias‑variance tradeoff, overfitting, evaluation metrics (F1, ROC‑AUC, FID)  
- Causal inference basics & Bayesian methods for small data

### Memory gems
- **p‑value** = “Proof value” – small means strong evidence  
- **Precision** = PREdict positive correctly; **Recall** = RECover actual positives  
- **KL divergence** = “Knowledge Loss”

---

## 3. Calculus I – Single Variable

**File:** [`calculus1_single_variable.md`](calculus1_single_variable.md)

### What you'll learn
- Limits – approaching a value without touching  
- Derivative as instantaneous rate of change (speedometer)  
- Differentiation rules (product, quotient, chain)  
- The Chain Rule – the secret sauce of backpropagation  
- Optimization: finding minima with gradient descent  
- Learning rate tuning and common activation derivatives (sigmoid, ReLU)  
- Integration as accumulation (expected value, marginal likelihood)

### Memory gems
- **Derivative** = “Divide by h, let h shrink”  
- **Chain rule** = “Chug along – outer′ × inner′”  
- **Gradient descent** = “Step opposite the slope”

---

## 4. Calculus II – Multivariable

**File:** [`calculus2_multivariable.md`](calculus2_multivariable.md)

### What you'll learn
- Functions of several variables (the loss landscape)  
- Partial derivatives – change one thing at a time  
- The Gradient – vector of steepest ascent (and why we go negative)  
- Directional derivatives – any direction you choose  
- Multivariate optimization: minima, maxima, saddle points  
- Gradient descent in high dimensions (SGD, momentum, Adam)  
- Lagrange multipliers for constrained optimization (regularization, SVMs)  
- Double integrals – marginalisation in probabilistic models

### Memory gems
- **Partial derivative** = “Part – change only one part”  
- **Gradient** = “Go Rampant In Direction of Extreme Tallness” (uphill)  
- **Saddle point** = “Up one way, down another”

---

## 5. Calculus III – Vector Calculus

**File:** [`calculus3_vector_calculus.md`](calculus3_vector_calculus.md)

### What you'll learn
- Vector‑valued functions & vector fields (wind maps)  
- The Jacobian matrix – derivative for vector functions (used in backprop)  
- Hessian matrix & curvature (Newton’s method, Fisher information)  
- Divergence, curl, and Laplacian (score‑based diffusion models)  
- Line integrals – work along a path (policy gradient theorem)  
- Backpropagation as the vector chain rule (Jacobian transposes)  
- Score functions – the gradient of log probability (diffusion models)  
- Policy gradients & natural gradients in reinforcement learning  
- Matrix calculus – differentiating with respect to weight matrices

### Memory gems
- **Jacobian** = “Just All Combinations” (matrix of partials)  
- **Divergence** = “Dispersal” – net outflow  
- **Score function** = gradient of log probability

---

## 🧠 How to Use This Series

### Recommended order
1. **Linear Algebra** – foundational (vectors → matrices → SVD)  
2. **Calculus I** – derivatives & chain rule  
3. **Statistics** – probability & inference  
4. **Calculus II** – multivariable & gradient  
5. **Calculus III** – vector & matrix calculus

> *You can study Statistics and Calculus II in parallel – they meet at MLE and Bayesian inference.*

### Each course contains:
- **Analogies** – real‑world comparisons  
- **Memory techniques** – forget‑proof mnemonics  
- **Code examples** – executable Python (NumPy, SciPy, PyTorch)  
- **Quizzes & exercises** – with answers  
- **Applications** – show exactly where the math appears in ML/DL/GenAI/Agentic AI

### Prerequisites
- High school mathematics (basic algebra + functions)  
- No calculus or linear algebra required – everything is built from scratch  
- Basic Python is helpful but not mandatory for the theory sections

---

## 🛠️ Tools & Libraries Used

```python
import numpy as np
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt
