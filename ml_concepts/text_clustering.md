## 🧩 What Is Text Clustering?

**Text clustering** is the process of grouping similar pieces of text (documents, sentences, tweets, etc.) based on their content, **without any predefined labels**.

---

## 🧪 Examples of Use Cases

| Use Case | Description |
|----------|-------------|
| **News categorization** | Automatically group similar news articles (e.g., politics, sports). |
| **Customer feedback** | Cluster survey responses to identify major themes. |
| **Topic modeling** | Group research abstracts into topics. |
| **Spam detection** | Cluster emails/messages to find spam-like patterns. |
| **FAQ detection** | Cluster user questions to match with existing answers. |

---

## 🧠 Common Methods

### 1. **Preprocessing**
- Tokenization, lowercasing
- Stopword removal
- Stemming/Lemmatization (optional)
- Convert text into vectors

### 2. **Vectorization**
- **TF-IDF** (bag-of-words with frequency scaling)
- **Word2Vec**, **GloVe** (use averaged embeddings)
- **Sentence embeddings**:  
  - `Sentence-BERT (SBERT)`
  - `Universal Sentence Encoder (USE)`
  - `OpenAI/transformer embeddings`

### 3. **Clustering Algorithms**

| Algorithm     | Description |
|---------------|-------------|
| **K-Means**   | Partitions text into _k_ clusters based on centroid similarity. |
| **DBSCAN**    | Density-based, great for discovering outliers/noise. |
| **Agglomerative** | Builds a tree of clusters (dendrogram). |
| **HDBSCAN**   | Hierarchical version of DBSCAN. |
| **Spectral Clustering** | Graph-based; good for non-convex shapes. |

---

## 📊 Evaluation Metrics

Since clustering is **unsupervised**, evaluation is tricky. You can use:

### If You Have Ground Truth (semi-supervised testing):
- **ARI (Adjusted Rand Index)**
- **NMI (Normalized Mutual Information)**
- **Fowlkes–Mallows Index**

### Without Ground Truth (purely unsupervised):
- **Silhouette Score**: Measures how similar a point is to its own cluster vs others.
- **Calinski-Harabasz Index**
- **Davies-Bouldin Index**

---

## 🧪 Example: Text Clustering with TF-IDF + K-Means

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

docs = [
    "I love machine learning",
    "Deep learning is part of AI",
    "Football is a great sport",
    "AI is transforming the world",
    "I like watching soccer games"
]

# Convert to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Cluster
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# View results
for doc, label in zip(docs, labels):
    print(f"Cluster {label}: {doc}")
```

---

## 🌍 Advanced: Semantic Clustering with Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(embeddings)
```

This version captures **context and meaning**, not just word overlap.
