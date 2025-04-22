 \)## 🧠 Recap Table

| Concept                               | Formula (Core)                                         | Interpretation                          | ML Use Case                             |
|----------------------------------------|----------------------------------------------------------|------------------------------------------|------------------------------------------|
| **Entropy (Shannon)** $H(P)$             | $-\sum P(x) \log P(x)$                                  | Uncertainty                              | Feature selection, information theory     |
| **Cross-Entropy** $H(P, Q)$              | $-\sum P(x) \log Q(x)$                                  | Surprise from wrong distribution         | Loss for classification                  |
| **Relative Entropy (KL Divergence)** $D_{\text{KL}}$ | $\sum P(x) \log \frac{P(x)}{Q(x)}$                      | Extra bits from using $Q$            | VAEs, RL, distillation                   |
| **Perplexity**                            | $2^{H(P)}$ or $\exp(H(P))$                         | Effective number of choices              | Language models, evaluation              |
| **Conditional Entropy** $H(X \mid Y)$    | $-\sum P(x, y) \log P(x \mid y)$                        | Uncertainty remaining given $Y$      | Info theory, generative modeling         |
| **Mutual Information** $I(X; Y)$        | $H(X) - H(X \mid Y)$                                   | Shared information between variables     | Feature selection, self-supervised learning |
| **Differential Entropy**                  | $-\int p(x) \log p(x) \, dx$                           | Continuous version of entropy            | Latent variable modeling                 |

---

## 🌪️ 1. **Entropy (Shannon Entropy)**

### ✅ Definition
Entropy measures the average amount of **uncertainty** or **information content** in a probability distribution.

### 📐 Formula
For a discrete probability distribution $P$ over outcomes $x \in \mathcal{X}$:

$$H(P) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)$$

- Base-2 log gives units in **bits**.

### 💡 Intuition
- High entropy = very uncertain (e.g., uniform distribution).
- Low entropy = very certain (e.g., one outcome has $P(x) = 1$).

### 📍 ML Use
- Feature selection: Information Gain = reduction in entropy.
- Decision trees: Choose features that reduce entropy the most.
- Language modeling: Measure the inherent uncertainty in word distributions.

---

## 🎯 2. **Cross-Entropy**

### ✅ Definition
Cross-entropy measures how different two probability distributions are: the **true distribution** $P$ and the **predicted distribution** $Q$.

### 📐 Formula
$$H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x)$$

### 💡 Intuition
“How surprised would I be on average if I thought outcomes followed $Q$ but the truth was $P$?”

### 📍 ML Use
- **Loss function** in classification tasks:
  - $P$: true labels (usually one-hot).
  - $Q$: predicted probabilities (softmax output).
- Minimizing cross-entropy encourages $Q \to P$, i.e., good predictions.

---

## 🔁 3. **Relative Entropy (Kullback–Leibler Divergence)**

### ✅ Definition
KL divergence measures how much one distribution $Q$ diverges from a true distribution $P$.

### 📐 Formula
$$D_{\text{KL}}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

### 💡 Intuition
- “Extra” bits you pay using the wrong distribution $Q$ instead of the true one $P$.
- Not symmetric: $D_{\text{KL}}(P \| Q) \ne D_{\text{KL}}(Q \| P)$.

### 📍 ML Use
- VAEs (Variational Autoencoders): KL divergence between latent distributions.
- Policy learning in RL.
- Model distillation: matching student and teacher distributions.

---

## 🔮 4. **Perplexity**

### ✅ Definition
Perplexity is the **exponential** of entropy: how **confused** a model is.

### 📐 Formula
$$\text{Perplexity}(P) = 2^{H(P)} = \exp(H(P)) \quad \text{(depending on log base)}$$

### 💡 Intuition
- "How many equally likely choices would give the same uncertainty?"
- If entropy is 3 bits, perplexity is $2^3 = 8$ → like choosing randomly among 8 things.

### 📍 ML Use
- NLP: Evaluate language models (lower perplexity = better).
- Compare how well a model compresses/represents data.

---

## 🔃 5. **Conditional Entropy**

### ✅ Definition
How much uncertainty remains about $X$ given $Y$?

### 📐 Formula
$$H(X|Y) = \sum_{y \in \mathcal{Y}} P(y) H(X|Y=y)
= -\sum_{x, y} P(x, y) \log P(x|y)$$

### 💡 Intuition
- Lower conditional entropy means $Y$ is informative about $X$.

### 📍 ML Use
- Mutual information:
  $$  I(X; Y) = H(X) - H(X|Y)$$
  
  Often used for feature selection and generative modeling.

---

## 🔗 6. **Mutual Information**

### ✅ Definition
How much information does one variable tell us about another?

### 📐 Formula
$$I(X; Y) = \sum_{x,y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)} = D_{\text{KL}}(P(X,Y) \| P(X)P(Y))$$

### 💡 Intuition
- Mutual information = reduction in uncertainty of $X$ due to knowing $Y$.
- $= 0$ if $X \perp Y$ (independent).

### 📍 ML Use
- Feature selection, InfoGAN, InfoBERT.
- Clustering and representation learning.

---

## 🧩 7. **Entropy in Continuous Distributions (Differential Entropy)**

### 📐 Formula
$$h(X) = -\int p(x) \log p(x) \, dx$$

### ❗Note
- Can be negative!
- Not invariant under change of variables.

### 📍 ML Use
- Latent variables in VAEs.
- Variational inference.

---

## 💣 Bonus: Max-Entropy Principle

### ✅ Idea
Choose the distribution with the **highest entropy** subject to known constraints.

### 📍 ML Use
- Used in logistic regression, exponential family modeling.
- Foundational in probabilistic modeling.

