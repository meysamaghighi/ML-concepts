## 📌 Classification Algorithms

| #  | Model                  | Type              | Description                                           | Pros                                               | Cons                                              | Example Code                                    | Notes                             |
|----|------------------------|-------------------|-------------------------------------------------------|----------------------------------------------------|---------------------------------------------------|--------------------------------------------------|-----------------------------------|
| 1  | [Naive Bayes](#1-naive-bayes)            | Probabilistic     | Probabilistic model for discrete features            | Simple, fast, great for text                       | Strong independence assumption                    | `MultinomialNB()` (sklearn)                      | Use with bag-of-words or TF-IDF   |
| 2  | [Logistic Regression](#2-logistic-regression)    | Linear Model      | Linear model for classification                      | Interpretable, fast                                | Poor with non-linear boundaries                   | `LogisticRegression()` (sklearn)                | Add regularization (L1/L2)        |
| 3  | [Decision Tree](#3-decision-tree)          | Tree-Based        | Tree structure based on feature splits               | Interpretable, no scaling needed                   | Easily overfits                                   | `DecisionTreeClassifier()` (sklearn)            | Basis for ensembles               |
| 4  | [Random Forest](#4-random-forest)          | Ensemble (Tree)   | Ensemble of decision trees                           | Robust, handles non-linearities                    | Less interpretable                                | `RandomForestClassifier()` (sklearn)            | Good default model                |
| 5  | [ExtraTreesClassifier](#5-extratreesclassifier)   | Ensemble (Tree)   | More randomized forest                               | Faster than RF, good for high-dim data             | May overfit small data                            | `ExtraTreesClassifier()` (sklearn)              | Try when RF is slow               |
| 6  | [XGBoost](#6-xgboost)                | Ensemble (Boost)  | Extreme gradient boosting                            | Accurate, fast, handles missing values             | Requires careful tuning                           | `XGBClassifier()` (xgboost)                     | Great on tabular data             |
| 7  | [LightGBM](#7-lightgbm)               | Ensemble (Boost)  | Efficient gradient boosting                          | Extremely fast, great on large datasets            | Requires numerical features                       | `LGBMClassifier()` (lightgbm)                   | GPU support                       |
| 8  | [CatBoost](#8-catboost)               | Ensemble (Boost)  | Handles categorical data natively                    | Great performance, minimal tuning                  | Slower training                                   | `CatBoostClassifier()` (catboost)               | No label encoding needed          |
| 9  | [SVC (SVM)](#9-svc-svm)              | Kernel Method     | Max-margin classifier with kernel trick              | Effective in high-dimensions                       | Slow on large data                                | `SVC(kernel='rbf')` (sklearn)                   | Use `LinearSVC` for large data    |
| 10 | [SGDClassifier](#10-sgdclassifier)          | Linear Model      | Online learning linear model                         | Very fast, handles streaming                       | Sensitive to scale                                | `SGDClassifier()` (sklearn)                     | Add regularization                |
| 11 | [KNN Classifier](#11-knn-classifier)         | Instance-Based    | Lazy classifier based on neighbors                   | No training, interpretable                         | Slow prediction                                   | `KNeighborsClassifier()` (sklearn)              | Normalize features                |
| 12 | [MLPClassifier](#12-mlpclassifier)          | Neural Network    | Neural network (feed-forward)                        | Captures complex patterns                          | Needs more data and tuning                        | `MLPClassifier(hidden_layer_sizes=(100,))`      | Sensitive to scaling              |
| 13 | [VotingClassifier](#13-votingclassifier)       | Ensemble (Vote)   | Combines multiple classifiers                        | Simple ensemble                                    | Needs similar scale models                        | `VotingClassifier(estimators=[...])` (sklearn)  | Hard or soft voting               |
| 14 | [StackingClassifier](#14-stackingclassifier)     | Ensemble (Stack)  | Meta-model on top of base classifiers                | Often best performance                             | More complexity                                   | `StackingClassifier(estimators=[...])` (sklearn)| Can overfit small datasets        |
| 15 | [QDA](#15-qda)                        | Probabilistic     | Quadratic Discriminant Analysis                    | Captures non-linear class boundaries              | Assumes Gaussian distribution per class          | `QuadraticDiscriminantAnalysis()` (sklearn)     | Use when classes are well-separated and Gaussian |
| 16 | [Gaussian Process Classifier](#16-gaussian-process-classifier) | Probabilistic     | Bayesian classifier using Gaussian Processes for class probability | Provides uncertainty estimates, flexible decision boundary | Computationally expensive, not scalable to large data       | `GaussianProcessClassifier()` (sklearn)              | Best for small datasets with uncertainty needs |
---

---

### 1. Naive Bayes
Naive Bayes is a simple yet powerful probabilistic classifier based on Bayes' theorem. It assumes strong independence between features, which rarely holds in practice but works surprisingly well in many scenarios, especially in Natural Language Processing (NLP). It calculates the posterior probability of each class given the features and selects the class with the highest probability.

**Intuition**: Imagine sorting emails into "spam" and "not spam" based on word frequency. Naive Bayes assumes that the presence of one word is independent of another, given the class.

**Math**: $$\left( P(C|X) \propto P(X|C)P(C) \right), where \left( X = (x_1, ..., x_n) \right).$$

**Example Code**:
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
```

**Use Case**: Text classification (spam detection, sentiment analysis), where input features are word frequencies or TF-IDF vectors.

---


### 2. Logistic Regression
Logistic Regression is a linear model used for binary (and multiclass) classification. It models the probability that a given input belongs to a particular category using the logistic (sigmoid) function. It's widely used because of its simplicity, speed, and interpretability.

**Intuition**: Logistic Regression learns a linear decision boundary and outputs probabilities. It works best when classes are linearly separable.

**Math**:
$$\left[ P(y=1|x) = \frac{1}{1 + e^{-w^T x}} \right]$$
The model is trained by minimizing the log-loss/cross-entropy (equivalently, maximizing the likelihood) between predicted and actual labels.

**Code Example**:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

Best used for linearly separable data and scenarios requiring model interpretability (e.g., in healthcare or finance). You can regularize it with L1 or L2 to avoid overfitting.

---

### 3. Decision Tree
Decision Trees split data based on feature thresholds to create a flowchart-like structure of decisions. Each internal node represents a decision rule, and each leaf node represents a class label.

**Intuition**: Trees recursively partition the feature space to create homogeneous subsets. Each split maximizes some criterion like information gain.

**Math**: Common split metrics:
- Gini Impurity: $\left( G = 1 - \sum p_i^2 \right)$
- Entropy: $\left( H = -\sum p_i \log(p_i) \right)$

**Code Example**:
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

Great for explainability and small datasets. Be cautious of overfitting — pruning, depth control, or ensemble methods can help.

---

### 4. Random Forest
Random Forest is an ensemble of Decision Trees trained on different bootstrap samples with feature randomness. Each tree votes and the majority wins.

**Intuition**: Reduces variance compared to a single tree by averaging predictions from many uncorrelated trees.

**Code Example**:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

A reliable and flexible model that performs well on a wide range of problems without much tuning.

---

### 5. ExtraTreesClassifier
Extra Trees (Extremely Randomized Trees) further randomizes the tree-building process compared to Random Forest by choosing split points at random.

**Intuition**: By injecting more randomness, Extra Trees aim to reduce variance and overfitting.

**Code Example**:
```python
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
```

Useful when training time is a concern or Random Forest is overfitting. Works well with high-dimensional datasets.

---

### 6. XGBoost
XGBoost (Extreme Gradient Boosting) is a powerful and scalable machine learning algorithm based on gradient boosting. It builds models sequentially, where each new model tries to correct the errors made by the previous one. It's known for its performance in Kaggle competitions.

**Features**:
- Regularization (L1/L2) built-in.
- Handles missing data internally.
- Supports early stopping and parallel processing.

**Intuition**: Combine many weak learners (shallow trees) to form a strong learner by minimizing a loss function iteratively.

**Math**: Adds trees by optimizing: $\left( L = \sum_{i} l(y_i, \hat{y}_i) + \sum_k \Omega(f_k) \right)$, where $\left( \Omega \right)$ is a regularization term.

**Example Code**:
```python
from xgboost import XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
```

**Use Case**: Tabular data for classification and regression tasks. Dominates structured data problems.

---

### 7. LightGBM
LightGBM (Light Gradient Boosting Machine) is designed for speed and efficiency. It uses histogram-based algorithms and grows trees leaf-wise, making it faster and more accurate on large datasets.

**Key Characteristics**:
- Grows trees leaf-wise instead of level-wise.
- Supports categorical features with proper encoding.
- GPU training support.

**Intuition**: By growing the leaf with the largest loss reduction, it achieves better accuracy but may overfit.

**Example Code**:
```python
from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(X_train, y_train)
```

**Use Case**: Large-scale ML tasks, ranking, and classification on tabular datasets.

---

### 8. CatBoost
CatBoost is a gradient boosting library by Yandex that natively handles categorical features. It automatically deals with categorical variables via target statistics, which eliminates the need for manual preprocessing.

**Advantages**:
- No need for one-hot encoding.
- Supports GPU and fast inference.
- High-quality default settings.

**Math**: Uses oblivious (symmetric) decision trees which help reduce overfitting and improve speed.

**Example Code**:
```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)
```

**Use Case**: Datasets with many categorical features. Performs well even without hyperparameter tuning.

---

### 9. SVC (SVM)
Support Vector Machines (SVM) are powerful classifiers that aim to find the optimal hyperplane that separates classes by the maximum margin. With the kernel trick, they can handle non-linear data.

**Kernel Trick**: Allows mapping data to higher-dimensional spaces without explicitly computing the transformation, enabling linear separability.

**Mathematics**: Maximize $\left( \frac{2}{\|w\|} \right)$, subject to correct classification.

**Example Code**:
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
```

**Use Case**: High-dimensional datasets like image recognition. Use `LinearSVC` or other approximations for scalability.


---

### 10. SGDClassifier
Stochastic Gradient Descent (SGD) Classifier is a linear model optimized using online gradient descent. It's efficient for large-scale datasets and supports different loss functions (e.g., hinge, log).

**Intuition**: Rather than computing gradients over the entire dataset, SGD updates weights using one (or a few) samples at a time.

**Code Example**:
```python
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log_loss')
model.fit(X_train, y_train)
```

Ideal for high-dimensional sparse data, online learning, and streaming scenarios. Scaling features improves performance.

---

### 11. KNN Classifier
K-Nearest Neighbors is a non-parametric, instance-based learning algorithm. It classifies new instances by a majority vote of its k closest neighbors in the training set.

**Intuition**: It assumes that similar instances exist in close proximity.

**Code Example**:
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

Great for small datasets where decision boundaries are not well-defined. Scaling features is important due to distance-based calculations.

---

### 12. MLPClassifier
Multi-Layer Perceptron (MLP) is a feed-forward artificial neural network trained with backpropagation. It can model non-linear relationships and complex patterns.

**Intuition**: The network learns weights through multiple layers of neurons (input -> hidden -> output). Each neuron applies a non-linear activation function.

**Code Example**:
```python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
model.fit(X_train, y_train)
```

Use MLP when data patterns are complex and non-linear. Requires tuning (layers, activation, learning rate) and scaling.

---

### 13. VotingClassifier
VotingClassifier combines multiple models and makes predictions based on majority (hard) or averaged probabilities (soft).

**Intuition**: Aggregating diverse models can often yield better performance than any single one.

**Code Example**:
```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[('lr', model1), ('rf', model2)], voting='soft')
ensemble.fit(X_train, y_train)
```

Best when you have diverse models with complementary strengths. Input models should be well-calibrated if using soft voting.

---

### 14. StackingClassifier
StackingClassifier trains multiple base learners and a meta-model (often logistic regression) to learn how to best combine them. It's more flexible and powerful than voting.

**Intuition**: The meta-model learns from predictions of base models — combining their strengths.

**Code Example**:
```python
from sklearn.ensemble import StackingClassifier
ensemble = StackingClassifier(estimators=[('svc', model1), ('rf', model2)], final_estimator=LogisticRegression())
ensemble.fit(X_train, y_train)
```

Use when accuracy is key and you can afford the complexity. May overfit on small datasets — cross-validation is helpful.

---

### 15. QDA

Quadratic Discriminant Analysis (QDA) is a probabilistic classification model that, like LDA, assumes each class follows a Gaussian distribution. However, unlike LDA, QDA allows each class to have its own covariance matrix, enabling it to capture quadratic decision boundaries.

**Intuition**: Imagine you're trying to separate two clouds of points in space, but each cloud has a different shape or orientation. QDA finds a curved boundary (a quadratic surface) to separate them.

**Assumptions**:
- Features are normally distributed within each class.
- Covariance matrices can differ across classes.

**Mathematics**: QDA models the class-conditional densities as Gaussians and applies Bayes' theorem to compute posterior probabilities:

$$ P(y=k|x) ∝ π_k * N(x | μ_k, Σ_k) $$

Where `π_k` is the prior, and `N` is the multivariate normal density with class-specific mean `μ_k` and covariance `Σ_k`.

**Example Code**:
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
```

**Use Case**: Use QDA when the data is Gaussian but has class-specific variances. It performs well when classes are well-separated and follow different distributions.

---

### 16. Gaussian Process Classifier

Gaussian Process Classifier (GPC) extends Gaussian Processes to **classification** tasks by modeling a distribution over functions that map inputs to class probabilities. Unlike regression, classification requires mapping continuous latent functions to a discrete label space using a sigmoid or softmax function.

---

**Intuition**:  
Instead of predicting exact values (like in regression), GPC models the **latent function** and then **squashes it** through a logistic function to get a probability of class membership. This allows GPC to output not just the predicted class but also the **confidence** in that prediction.

> Think of it as: "At point `x`, what's the probability this belongs to class 1, given what we know about the training data and how similar `x` is to them?"

---

**Math**:
1. Define latent function \( f(x) \sim \mathcal{GP}(0, k(x, x')) \)
2. Use a **sigmoid** (e.g., logistic) link function to convert latent function to probability:
   \[
   p(y=1|x) = \sigma(f(x))
   \]
3. Use approximate inference (e.g., Laplace approximation or EP) since the posterior is non-Gaussian.

---

**Example Code**:
```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Create a toy dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# Define GP Classifier with RBF kernel
kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel)
gpc.fit(X, y)

# Plot decision boundary
import numpy as np
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
Z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=25, cmap='RdBu', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
plt.title("Gaussian Process Classifier Decision Boundary")
plt.show()
```

---

**Use Case**:
- Binary or multi-class classification with small datasets
- When **uncertainty quantification** is important (e.g., medical diagnosis, scientific experiments)
- Alternative to SVM when probabilistic output is preferred

