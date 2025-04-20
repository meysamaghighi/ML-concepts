# 🧠 Dimensionality Reduction Algorithms

Dimensionality reduction is essential for:
- Visualization of high-dimensional data
- Noise reduction
- Speeding up training
- Discovering latent patterns or structure

---

## 📋 Overview Comparison Table

| Method               | Description                                  | Pros                                       | Cons                                      | Notes                                               |
|----------------------|----------------------------------------------|--------------------------------------------|-------------------------------------------|------------------------------------------------------|
| **[PCA](#1-pca-principal-component-analysis)**      | Linear projection that maximizes variance    | Fast, interpretable                        | Linear only                                | Projects data along directions of max variance       | 
| **[Truncated SVD](#2-truncated-svd)** | SVD for sparse data                       | Works well with sparse matrices            | No mean centering                          | Used in LSA (text data)                              | 
| **[SVD](#3-svd-singular-value-decomposition)**      | General matrix factorization                 | Mathematically powerful                    | Not always interpretable                   | Used for reconstruction & compression                | 
| **[t-SNE](#4-t-sne-t-distributed-stochastic-neighbor-embedding)**   | Probabilistic method for local structure     | Great for visualization                    | Slow, non-deterministic                    | Doesn’t preserve global structure                    | 
| **[Autoencoders](#5-autoencoders)** | Neural net for data reconstruction      | Non-linear, customizable                   | Requires training, more resources          | Learns via backpropagation                           | 
| **[UMAP](#6-umap-uniform-manifold-approximation-and-projection)**    | Graph-based, faster alternative to t-SNE     | Preserves local & some global structure    | Sensitive to parameters                    | Excellent for visualization                          | 
| **[Isomap](#7-isomap)**| Manifold learning using geodesic distance    | Captures non-linear global structure       | Sensitive to noise and outliers            | Uses neighborhood graph distances                    | 
| **[LLE (Locally Linear Embedding)](#8-lle-locally-linear-embedding)**      | Preserves local linear neighborhood          | Simple non-linear manifold                 | Struggles with high noise                  | Learns local affine reconstructions                  | 
| **[Spectral Clustering](#9-spectral-clustering)** | Uses graph Laplacian eigenvectors | Clusters with complex boundaries           | Requires similarity graph tuning           | Great for non-convex clusters                        | 
| **[Matrix Factorization](#10-matrix-factorization)** | Low-rank approximation                  | Very effective in recommender systems      | Needs sparsity pattern                     | Popular in collaborative filtering                   | 
| **[LDA (Linear Discriminant Analysis)](#11-lda-linear-discriminant-analysis)**              | Supervised projection to separate classes    | Great for classification, interpretable    | Needs labeled data, class-count limit      | Maximizes class separability                         |

---

## 1. PCA (Principal Component Analysis) <a name="pca"></a>

### 🧠 Idea:
Finds orthogonal directions of maximum variance.

### 📐 Math:
```math
C = \frac{1}{n} X^T X \quad \text{(Covariance matrix)}
```
```math
X_{\text{reduced}} = X V_k
```

### 🐍 Python:
```python
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=2).fit_transform(X)
```

---

## 2. Truncated SVD 

### 🧠 Idea:
Performs dimensionality reduction without centering, good for sparse data.

### 📐 Math:
```math
X \approx U_k \Sigma_k V_k^T
```

### 🐍 Python:
```python
from sklearn.decomposition import TruncatedSVD
X_reduced = TruncatedSVD(n_components=2).fit_transform(X_sparse)
```

---

## 3. SVD (Singular Value Decomposition) 

### 🧠 Idea:
Factorizes matrix into orthogonal components.

### 📐 Math:
```math
X = U \Sigma V^T
```

### 🐍 Python:
```python
U, S, VT = np.linalg.svd(X, full_matrices=False)
X_reduced = U[:, :2] @ np.diag(S[:2])
```

---

## 4. t-SNE (t-distributed Stochastic Neighbor Embedding) 

### 🧠 Idea:
Preserves local similarity using probability distributions.

### 📐 Math:
```math
\text{KL}(P \| Q) = \sum p_{ij} \log \frac{p_{ij}}{q_{ij}}
```

### 🐍 Python:
```python
from sklearn.manifold import TSNE
X_reduced = TSNE(n_components=2).fit_transform(X)
```

---

## 5. Autoencoders 

### 🧠 Idea:
Neural networks trained to reconstruct input.

### 📐 Math:
```math
\mathcal{L} = \sum \|x_i - \hat{x}_i\|^2
```

### 🐍 Python (Keras):
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape=(X.shape[1],))
encoded = Dense(2, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=20, batch_size=128)
```

---

## 6. UMAP (Uniform Manifold Approximation and Projection) 

### 🧠 Idea:
Constructs a graph in high and low dimensions and minimizes the difference.

### 📐 Math:
Optimizes cross-entropy between fuzzy simplicial sets.

### 🐍 Python:
```python
import umap
X_reduced = umap.UMAP(n_components=2).fit_transform(X)
```

---

## 7. Isomap

### 🧠 Idea:
Preserves geodesic distances using a neighborhood graph.

### 📐 Math:
1. Build graph
2. Use MDS on shortest path distances

### 🐍 Python:
```python
from sklearn.manifold import Isomap
X_reduced = Isomap(n_components=2).fit_transform(X)
```

---

## 8. LLE (Locally Linear Embedding) 

### 🧠 Idea:
Each point is a linear combination of neighbors in both high and low dimensions.

### 📐 Math:
Minimizes:
```math
\sum_i \|x_i - \sum_j w_{ij} x_j\|^2
```

### 🐍 Python:
```python
from sklearn.manifold import LocallyLinearEmbedding
X_reduced = LocallyLinearEmbedding(n_components=2).fit_transform(X)
```

---

## 9. Spectral Clustering 

### 🧠 Idea:
Clusters based on Laplacian eigenvectors of similarity graph.

### 📐 Math:
```math
L = D - W, \quad L_{\text{sym}} = D^{-1/2} L D^{-1/2}
```

### 🐍 Python:
```python
from sklearn.cluster import SpectralClustering
labels = SpectralClustering(n_clusters=3).fit_predict(X)
```

---

## 10. Matrix Factorization 

### 🧠 Idea:
Approximates a matrix by product of two low-rank matrices.

**Matrix Factorization** refers to techniques where a matrix is approximated as a product of two (or more) smaller matrices. In recommender systems, this is often done to **learn latent features** from user–item interactions.

### Example Use Case:
Imagine a **user-item rating matrix** $R \in \mathbb{R}^{n \times m}$ (users × items), where many entries are **missing** (i.e., unrated items).

Matrix Factorization aims to find two low-rank matrices:
```math
R \approx U V^T
```
- $U \in \mathbb{R}^{n \times k}$ (user features)
- $V \in \mathbb{R}^{m \times k}$ (item features)
- $k \ll n, m$ (latent dimension)

---

## 📐 Optimization Objective

We only care about the **observed entries** in $R$, so the optimization is:

```math
\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i^T V_j)^2 + \lambda(\|U_i\|^2 + \|V_j\|^2)
```

Where:
- $\Omega$ = set of observed (user, item) pairs
- $\lambda$ = regularization to avoid overfitting

---

## 🔄 Matrix Factorization vs SVD vs Truncated SVD

| Feature                | Matrix Factorization           | SVD                          | Truncated SVD                     |
|------------------------|-------------------------------|------------------------------|-----------------------------------|
| Handles missing data   | ✅ Yes                         | ❌ No                         | ❌ No                              |
| Learns latent factors  | ✅ Yes                         | ✅ Yes                        | ✅ Yes                             |
| Input type             | Sparse with missing values     | Dense, complete matrix       | Sparse, but still complete matrix |
| Centering required     | ❌ Not necessary               | ✅ Required                   | ❌ Not required                    |
| Output matrices        | $U$, $V$ (learned)             | $U$, $\Sigma$, $V^T$ (exact) | Approximate $U_k \Sigma_k V_k^T$ |
| Optimization           | Gradient descent / ALS         | Analytical decomposition     | Analytical (with dimensionality cap) |
| Use case               | Recommender systems            | Compression, topic modeling  | Text, NLP (LSA), compression      |

---

## 💡 Why Is Matrix Factorization Great for Recommenders?

1. **Latent factors**: It learns underlying patterns (e.g., a user likes action movies, an item is romantic).
2. **Fills in the blanks**: Predicts ratings for unseen user-item pairs.
3. **Scalable**: Efficient with large, sparse matrices.
4. **Personalized**: Embeds each user and item into a shared latent space.
5. **Real-world use**: Netflix, Amazon, Spotify used it in early collaborative filtering.

---

### 🐍 Example with `surprise` (scikit-like API for recommenders)

```python
from surprise import SVD, Dataset, Reader
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

# Predict rating for user 1 and item 302
pred = algo.predict(uid='1', iid='302')
print(pred.est)
```

This version of SVD is actually a **matrix factorization model** optimized over observed entries — not classic full-matrix SVD.

---

## 🔁 TL;DR

| Question                     | Answer |
|-----------------------------|--------|
| What is it?                 | Low-rank approximation of observed matrix with missing entries |
| How is it different from SVD? | Handles missing values, optimized via gradient descent |
| Why good for recommenders?  | Learns personalized user/item embeddings; scalable and accurate |

---

## 11. LDA (Linear Discriminant Analysis) 

### 🧠 Idea:
LDA is a **supervised** dimensionality reduction technique that projects data onto axes that best **separate known class labels**.

It’s different from PCA in that **LDA uses class information**, while PCA ignores it.

---

### 📐 Math:

LDA tries to **maximize the ratio** of **between-class variance** to **within-class variance**:

```math
\text{argmax}_W \frac{|W^T S_B W|}{|W^T S_W W|}
```

Where:
- $S_B$: **Between-class scatter** matrix
- $S_W$: **Within-class scatter** matrix
- $W$: projection matrix

You project the input as:

```math
X_{\text{reduced}} = X W
```

---

### 🐍 Python Example:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_reduced = lda.fit_transform(X, y)  # y are class labels
```

---

### ✅ Notes:
- Works only if you have **labels**
- Maximum number of components is **(number of classes - 1)**
- Often used before classifiers like SVM, Logistic Regression

---


