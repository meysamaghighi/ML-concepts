### 🔷 Overview of Learning Paradigms in ML

Machine Learning paradigms define **how a model learns from data** depending on the **type of supervision** (i.e., labels). There are five main paradigms:

1. **Supervised Learning**
2. **Unsupervised Learning**
3. **Semi-Supervised Learning**
4. **Self-Supervised Learning**
5. **Reinforcement Learning**

Let’s dive into each one.

---

### 1. 🧠 Supervised Learning

#### ✔️ Core Idea:
Learn a function **f(x) = y** from input-output pairs \((x_i, y_i)\).

#### ✔️ Training Data:
Labeled data — every input \(x_i\) has a corresponding label \(y_i\).

#### ✔️ Objective:
Minimize the prediction error (loss) between predicted \( \hat{y}_i \) and true \( y_i \).

#### ✔️ Common Loss:
- Classification: Cross-entropy loss
- Regression: Mean Squared Error (MSE)

#### ✔️ Example:
- Email spam detection
- Image classification (e.g., "cat" vs. "dog")

#### ✔️ Formal:
\[
\min_{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i)
\]

---

### 2. 🔍 Unsupervised Learning

#### ✔️ Core Idea:
Discover hidden patterns or structures in **unlabeled data**.

#### ✔️ Training Data:
Only input features \( x_i \), no labels.

#### ✔️ Objective:
Group similar inputs or reduce dimensionality.

#### ✔️ Common Algorithms:
- Clustering (K-Means, DBSCAN)
- Dimensionality reduction (PCA, t-SNE, autoencoders)

#### ✔️ Example:
- Customer segmentation
- Topic modeling in text

#### ✔️ Formal (e.g., clustering):
Find cluster centers \( \mu_1, \dots, \mu_k \) minimizing intra-cluster variance:
\[
\min \sum_{i=1}^N \min_{j=1, \dots, k} \| x_i - \mu_j \|^2
\]

---

### 3. 🌓 Semi-Supervised Learning

#### ✔️ Core Idea:
Use a small amount of labeled data and a large amount of unlabeled data.

#### ✔️ Motivation:
Labeling is expensive; unlabeled data is abundant.

#### ✔️ How It Works:
- Use labeled data to bootstrap initial learning
- Use model predictions on unlabeled data to refine or regularize the training

#### ✔️ Techniques:
- Pseudo-labeling: label unlabeled data with current model predictions
- Consistency regularization: enforce that predictions are stable under perturbations

#### ✔️ Example:
- Classifying documents when only a few are manually tagged

#### ✔️ Formal (sketch):
\[
\min_\theta \left[ \sum_{(x_i, y_i) \in \mathcal{D}_L} \mathcal{L}(f_\theta(x_i), y_i) + \lambda \sum_{x_j \in \mathcal{D}_U} \mathcal{R}(f_\theta(x_j)) \right]
\]
Where \( \mathcal{R} \) is a regularizer on unlabeled data.

---

### 4. 🧩 Self-Supervised Learning

#### ✔️ Core Idea:
Create **pretext tasks** from unlabeled data where labels are generated **automatically from the data itself**.

#### ✔️ Motivation:
Avoid manual labeling and pre-train useful representations.

#### ✔️ How It Works:
Design a supervised task (e.g., predict part of input from another part) to learn representations, which are later fine-tuned on real tasks.

#### ✔️ Examples:
- In NLP: Masked Language Modeling (e.g., BERT — predict masked words)
- In Vision: SimCLR / BYOL — contrastive learning using data augmentations

#### ✔️ Key Term: **Contrastive Learning**
Encourages representations \( f(x) \) of similar inputs to be close, and dissimilar ones far.

#### ✔️ Formal (contrastive):
\[
\mathcal{L} = -\log \frac{\exp(\text{sim}(f(x), f(x^+)) / \tau)}{\sum_{x^-} \exp(\text{sim}(f(x), f(x^-)) / \tau)}
\]
Where:
- \( x^+ \): positive (same instance, different view)
- \( x^- \): negatives (different instances)
- \( \text{sim} \): cosine similarity

---

### 5. 🧮 Reinforcement Learning

#### ✔️ Core Idea:
An **agent** learns to **take actions** in an environment to **maximize cumulative reward**.

#### ✔️ Key Components:
- State \( s \)
- Action \( a \)
- Reward \( r \)
- Policy \( \pi(a|s) \)

#### ✔️ Objective:
Maximize expected return:
\[
\max_\pi \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t r_t \right]
\]
Where \( \gamma \in [0, 1] \) is a discount factor.

#### ✔️ Examples:
- Game-playing agents (AlphaGo)
- Robotics

---

### 🧠 Comparison Summary Table

| Paradigm               | Labels?            | Example Task                        | Key Method                    |
|------------------------|--------------------|-------------------------------------|-------------------------------|
| Supervised             | ✅ Full             | Image classification                | Logistic regression, ResNet   |
| Unsupervised           | ❌ None             | Customer segmentation               | K-Means, PCA                  |
| Semi-Supervised        | ⚡ Few + many       | Classify few-labeled documents      | Pseudo-labeling, MixMatch     |
| Self-Supervised        | ❌ None (internal)  | Pre-train vision/text models        | Contrastive loss, MLM         |
| Reinforcement Learning | 🎯 Reward signal    | Play chess, robot control           | Q-learning, Policy gradient   |

---

### 🔚 Summary

- **Supervised**: Learn directly from labeled data.
- **Unsupervised**: Discover structure in unlabeled data.
- **Semi-Supervised**: Combine both to learn better with less labeling.
- **Self-Supervised**: Learn from pretext tasks to bootstrap representations.
- **Reinforcement**: Learn from interaction and delayed feedback.

---
