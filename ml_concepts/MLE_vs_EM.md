## 🧠 What is MLE?

**Maximum Likelihood Estimation (MLE)** is a method for estimating the parameters of a statistical model by finding the values that **maximize the likelihood** of the observed data.

### 👇 Simple idea:
Given data $X$ and a model with parameters $\theta$,  
MLE finds:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(X | \theta)$$

You use optimization techniques to directly **maximize this likelihood**.

---

## 🔮 But... What if We Have Hidden or Missing Data?

This is where **MLE struggles**.

Let’s say you observe $X$, but there are **latent/unobserved variables** $Z$ (e.g., cluster assignments in a mixture model).

Now the likelihood becomes:

$$P(X | \theta) = \sum_Z P(X, Z | \theta)$$

That **sum can be intractable** to compute or optimize directly.  
💡 So instead of maximizing $P(X|\theta)$ directly, we use…

---

## 🔁 What is Expectation-Maximization (EM)?

**EM is an iterative algorithm to do MLE in the presence of latent variables.**

---

### ✅ High-level Steps:

Let’s say we observe $X$, and there are hidden variables $Z$, and our model has parameters $\theta$.

1. **E-step (Expectation):**  
   Estimate the distribution over hidden variables $Z$ given current parameters $\theta^{(t)}$:

   $$Q(Z) = P(Z | X, \theta^{(t)})$$

2. **M-step (Maximization):**  
   Maximize the expected complete log-likelihood:

   $$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{Z \sim Q(Z)}[\log P(X, Z | \theta)]$$

3. **Repeat until convergence.**

---

## 📦 Real Example: Gaussian Mixture Models (GMM)

In GMM:
- You observe data $X$
- Latent variable $Z$: which Gaussian component each point belongs to
- You want to learn the means, variances, and mixing proportions

MLE is hard because you’d need to marginalize over all possible assignments $Z$.  
EM makes this tractable:
- **E-step:** compute soft cluster assignments
- **M-step:** update means, variances, and weights

---

## 🥊 EM vs. MLE: Comparison Table

| Feature | MLE | EM |
|--------|-----|----|
| Goal | Maximize likelihood | Maximize likelihood (with latent variables) |
| Handles hidden/missing data? | ❌ No | ✅ Yes |
| Optimization | Direct | Iterative (E-step + M-step) |
| Needs conditional probabilities? | ❌ No | ✅ Yes (e.g., $P(Z|X, \theta)$) |
| Used in | Linear/logistic regression, etc. | GMMs, HMMs, LDA, incomplete data problems |
| Convergence | May be faster | Slower but structured |

---

## 🎯 Key Intuition

MLE:
> "I know everything I need — I’ll just find the best parameters to fit my data."

EM:
> "Some of the data is hidden — I’ll **guess** the hidden parts, then **optimize**, then repeat until I’m confident."

