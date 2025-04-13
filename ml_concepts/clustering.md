## 📌 Unsupervised Learning Models

| #  | Model                                                                                     | Type                    | Description                          | Pros                                 | Cons                                | Example Code                             | Notes                                    |
|----|--------------------------------------------------------------------------------------------|-------------------------|--------------------------------------|--------------------------------------|-------------------------------------|------------------------------------------|------------------------------------------|
| 1  | [KMeans](#1-kmeans)                                                                       | Clustering              | Centroid-based clustering            | Simple, scalable                     | Assumes spherical clusters          | `KMeans()` (sklearn)                     | Use `k-means++` init                      |
| 2  | [MiniBatchKMeans](#2-minibatchkmeans)                                                     | Clustering              | Faster version of KMeans             | Scales to big data                   | Less accurate                       | `MiniBatchKMeans()` (sklearn)            | Use for streaming data                    |
| 3  | [DBSCAN](#3-dbscan)                                                                       | Clustering              | Density-based clusters               | Finds arbitrary shapes               | Bad with varying densities          | `DBSCAN()` (sklearn)                     | Doesn’t require n_clusters                |
| 4  | [OPTICS](#4-optics)                                                                       | Clustering              | Advanced density clustering          | Handles varying density              | More complex to interpret           | `OPTICS()` (sklearn)                     | Outputs reachability graph                |
| 5  | [HDBSCAN](#5-hdbscan)                                                                     | Clustering              | Hierarchical DBSCAN                  | Very good for real-world data        | External library                    | `HDBSCAN()` (hdbscan)                    | Better than DBSCAN                        |
| 6  | [Agglomerative](#6-agglomerative)                                                         | Clustering              | Bottom-up hierarchical clustering    | Simple, visualizable                 | Not scalable                        | `AgglomerativeClustering()` (sklearn)    | Use dendrogram                            |
| 7  | [Birch](#7-birch)                                                                         | Clustering              | Hierarchical clustering on large data| Efficient, scalable                  | May miss global structure           | `Birch()` (sklearn)                      | Use as pre-step                           |
| 8  | [Spectral Clustering](#8-spectral-clustering)                                             | Clustering              | Graph-based method                   | Works on non-convex data             | Slow, needs full similarity matrix  | `SpectralClustering()` (sklearn)         | Good for few clusters                     |
| 9  | [MeanShift](#9-meanshift)                                                                 | Clustering              | Sliding window density clustering    | No need to predefine clusters        | Very slow                           | `MeanShift()` (sklearn)                  | Good on smooth distributions              |
| 10 | [Gaussian Mixture Model](#10-gaussian-mixture-model)                                      | Clustering              | Probabilistic soft clustering        | Captures overlapping clusters        | Needs to estimate #components       | `GaussianMixture()` (sklearn.mixture)    | Soft cluster assignments                  |
| 11 | [Affinity Propagation](#11-affinity-propagation)                                          | Clustering              | Exemplar-based clustering            | No need to specify clusters          | Very memory-intensive               | `AffinityPropagation()` (sklearn)        | Slow on large datasets                    |
| 12 | [Self-Organizing Maps (SOM)](#12-self-organizing-maps-som)                                | Clustering / Neural     | Neural map preserving topology       | Good for visualization               | Requires tuning, not in sklearn     | `MiniSom` (external lib)                 | Used for 2D projection + clustering       |
| 13 | [Expectation Maximization (EM)](#13-expectation-maximization-em)                          | Probabilistic Optimization| Iterative method to fit GMMs       | Soft assignments, flexible           | Can converge to local optima        | Built into `GaussianMixture()`           | Used inside GMM, HMM, etc.                |
| 14 | [Isolation Forest](#14-isolation-forest)                                                  | Anomaly Detection       | Random splits isolate outliers       | Scalable, handles high dimensions    | Not good for small datasets         | `IsolationForest()` (sklearn)            | Good default for anomaly detection        |
| 15 | [One-Class SVM](#15-one-class-svm)                                                        | Anomaly Detection       | Learns boundary around normal class  | Effective in high-dimensions         | Sensitive to scale, slow on big data| `OneClassSVM()` (sklearn)                | Use with normalized data                  |
| 16 | [Local Outlier Factor (LOF)](#16-local-outlier-factor-lof)                                | Anomaly Detection       | Compares local density to neighbors  | No training phase, intuitive         | Not good for extrapolation          | `LocalOutlierFactor()` (sklearn)         | Good for local anomaly detection          |
| 17 | [Markov Methods](#17-markov-methods)                                                      | Sequence Modeling       | Transition-based temporal modeling   | Powerful for sequences               | Assumes Markov property             | `hmmlearn`, `pomegranate`                | Used in HMMs, time series                 |
| 18 | [Deep Belief Nets (DBN)](#18-deep-belief-nets-dbn)                                        | Deep Unsupervised       | Layer-wise generative neural net     | Learns features unsupervised         | Obsolete vs modern deep learning    | `nolearn.dbn`, or PyTorch/TensorFlow     | Used before autoencoders/VAEs             |


---

### 1. **KMeans**
- **Math**: Minimizes sum of squared distances within each cluster (i.e., how far points are from the cluster center).
- **Idea**: Assign each point to its nearest center → move centers to the mean of their points → repeat.
- **Code**:
  ```python
  from sklearn.cluster import KMeans
  model = KMeans(n_clusters=3).fit(X)
  ```
- **Use Case**: Segmenting customers into groups, reducing image colors.

---

### 2. **MiniBatchKMeans**
- **Math**: Same as KMeans, but uses small random subsets of data (mini-batches) to update centers.
- **Idea**: Faster and uses less memory than full KMeans.
- **Code**:
  ```python
  from sklearn.cluster import MiniBatchKMeans
  model = MiniBatchKMeans(n_clusters=3).fit(X)
  ```
- **Use Case**: Clustering very large datasets in real time.

---

### 3. **DBSCAN**
- **Math**: Clusters dense regions, marks sparse points as noise.
- **Idea**: Points with enough close neighbors become clusters; isolated points don’t.
- **Code**:
  ```python
  from sklearn.cluster import DBSCAN
  model = DBSCAN(eps=0.5, min_samples=5).fit(X)
  ```
- **Use Case**: GPS data clustering, finding unusual patterns.

---

### 4. **OPTICS**
- **Math**: Orders points by how easily they can be reached (reachability).
- **Idea**: Like DBSCAN but can handle varying cluster densities.
- **Code**:
  ```python
  from sklearn.cluster import OPTICS
  model = OPTICS().fit(X)
  ```
- **Use Case**: More flexible density-based clustering.

---

### 5. **HDBSCAN**
- **Math**: Builds a tree of cluster stability, selects most stable clusters.
- **Idea**: Improves DBSCAN with a hierarchy and soft assignments.
- **Code**:
  ```python
  import hdbscan
  model = hdbscan.HDBSCAN().fit(X)
  ```
- **Use Case**: Text data, genetics, messy real-world clustering.

---

### 6. **Agglomerative Clustering**
- **Math**: Merge closest clusters step-by-step (bottom-up).
- **Idea**: Start with all points separate → merge closest → stop when desired cluster count is reached.
- **Code**:
  ```python
  from sklearn.cluster import AgglomerativeClustering
  model = AgglomerativeClustering(n_clusters=3).fit(X)
  ```
- **Use Case**: Dendrogram visualization of hierarchy.

---

### 7. **Birch**
- **Math**: Builds a tree structure summarizing the dataset (CF tree).
- **Idea**: Efficiently clusters data in one pass using compact representations.
- **Code**:
  ```python
  from sklearn.cluster import Birch
  model = Birch(n_clusters=3).fit(X)
  ```
- **Use Case**: Streaming or very large datasets.

---

### 8. **Spectral Clustering**
- **Math**: Converts data into a graph → uses eigenvectors of graph Laplacian.
- **Idea**: Good for non-circular shapes; sees clusters via graph cuts.
- **Code**:
  ```python
  from sklearn.cluster import SpectralClustering
  model = SpectralClustering(n_clusters=3).fit(X)
  ```
- **Use Case**: Clustering “moon”-shaped or intertwined data.

---

### 9. **MeanShift**
- **Math**: Moves points toward local data density peaks.
- **Idea**: Each point shifts until it reaches a high-density area → clusters form around those.
- **Code**:
  ```python
  from sklearn.cluster import MeanShift
  model = MeanShift().fit(X)
  ```
- **Use Case**: Image segmentation, finding modes.

---

### 10. **Gaussian Mixture Model (GMM)**
- **Math**: Uses EM (Expectation-Maximization) to fit multiple Gaussian distributions.
- **Idea**: Each point belongs to all clusters with some probability (soft clustering).
- **Code**:
  ```python
  from sklearn.mixture import GaussianMixture
  model = GaussianMixture(n_components=3).fit(X)
  ```
- **Use Case**: Anomaly detection, generative modeling.

---

### 11. **Affinity Propagation**
- **Math**: Passes messages between points to find exemplars (cluster centers).
- **Idea**: No need to specify number of clusters — the algorithm chooses them.
- **Code**:
  ```python
  from sklearn.cluster import AffinityPropagation
  model = AffinityPropagation().fit(X)
  ```
- **Use Case**: Clustering when you don’t know k.

---

### 12. **Self-Organizing Maps (SOM)**
- **Math**: Neural network grid that adjusts weights based on data proximity.
- **Idea**: Projects high-dimensional data onto a 2D grid while keeping structure.
- **Code**:
  ```python
  from minisom import MiniSom
  som = MiniSom(5, 5, len(X[0]))
  som.train(X, 100)
  ```
- **Use Case**: Visualizing and clustering high-dimensional data.

---

### 13. **Expectation Maximization (EM)**
- **Math**: Iterates between assigning probabilities (E) and updating parameters (M).
- **Idea**: Foundation of soft clustering — used in GMM and Hidden Markov Models.
- **Code**: Built into GMM and HMM libraries.
- **Use Case**: Clustering, time-series modeling.

---

### 14. **Isolation Forest**
- **Math**: Builds trees that isolate points by random splits.
- **Idea**: Outliers are isolated with fewer splits.
- **Code**:
  ```python
  from sklearn.ensemble import IsolationForest
  model = IsolationForest().fit(X)
  y_pred = model.predict(X)
  ```
- **Use Case**: Anomaly detection in logs or transactions.

---

### 15. **One-Class SVM**
- **Math**: Finds a boundary around normal data using a kernel function.
- **Idea**: Learns what’s "normal" and flags everything else as outliers.
- **Code**:
  ```python
  from sklearn.svm import OneClassSVM
  model = OneClassSVM(kernel='rbf', nu=0.1).fit(X)
  y_pred = model.predict(X)
  ```
- **Use Case**: Anomaly detection without outlier examples.

---

### 16. **Local Outlier Factor (LOF)**
- **Math**: Compares local density of a point to its neighbors.
- **Idea**: Points much less dense than neighbors are outliers.
- **Code**:
  ```python
  from sklearn.neighbors import LocalOutlierFactor
  model = LocalOutlierFactor(n_neighbors=20)
  y_pred = model.fit_predict(X)
  ```
- **Use Case**: Detecting anomalies in spatial/clustered data.

---

### 17. **Markov Methods (e.g., HMM)**
- **Math**: Models sequences with transition probabilities (P(next | current)).
- **Idea**: Future state depends only on the current state.
- **Code**:
  ```python
  from hmmlearn import hmm
  model = hmm.GaussianHMM(n_components=3).fit(X)
  ```
- **Use Case**: Speech recognition, market modeling.

---

### 18. **Deep Belief Nets (DBN)**
- **Math**: Stack of RBMs trained layer by layer.
- **Idea**: Learns features at multiple levels of abstraction.
- **Code**:
  ```python
  from nolearn.dbn import DBN
  model = DBN(...).fit(X, y)
  ```
- **Use Case**: Pre-training deep neural networks (historical use).

---

