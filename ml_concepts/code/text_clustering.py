from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd

# Sample documents
docs = [
    "I love machine learning",
    "Deep learning is part of AI",
    "Football is a great sport",
    "AI is transforming the world",
    "I like watching soccer games",
    "Neural networks power modern AI",
    "Soccer and football are popular sports",
    "Supervised learning is a key ML method",
    "The match was intense and exciting",
    "Natural language processing is fun"
]

# --- TF-IDF + KMeans ---
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(docs)
kmeans_tfidf = KMeans(n_clusters=3, random_state=42)
labels_tfidf = kmeans_tfidf.fit_predict(X_tfidf)
pca_tfidf = PCA(n_components=2)
reduced_tfidf = pca_tfidf.fit_transform(X_tfidf.toarray())
df_tfidf = pd.DataFrame({
    "x": reduced_tfidf[:, 0],
    "y": reduced_tfidf[:, 1],
    "label": labels_tfidf,
    "text": docs
})
silhouette_tfidf = silhouette_score(X_tfidf, labels_tfidf)
calinski_tfidf = calinski_harabasz_score(X_tfidf.toarray(), labels_tfidf)

# --- SBERT + KMeans ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)
scaled_embeddings = StandardScaler().fit_transform(embeddings)
kmeans_sbert = KMeans(n_clusters=3, random_state=42)
labels_sbert = kmeans_sbert.fit_predict(scaled_embeddings)
pca_sbert = PCA(n_components=2)
reduced_sbert = pca_sbert.fit_transform(scaled_embeddings)
df_sbert = pd.DataFrame({
    "x": reduced_sbert[:, 0],
    "y": reduced_sbert[:, 1],
    "label": labels_sbert,
    "text": docs
})
silhouette_sbert = silhouette_score(scaled_embeddings, labels_sbert)
calinski_sbert = calinski_harabasz_score(scaled_embeddings, labels_sbert)

# --- Plot with metrics ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# TF-IDF Plot
for label in df_tfidf['label'].unique():
    subset = df_tfidf[df_tfidf['label'] == label]
    axes[0].scatter(subset['x'], subset['y'], label=f"Cluster {label}")
for _, row in df_tfidf.iterrows():
    axes[0].text(row['x'] + 0.01, row['y'] + 0.01, row['text'], fontsize=8)
axes[0].set_title(f"TF-IDF + KMeans\nSilhouette: {silhouette_tfidf:.2f}, CH Index: {calinski_tfidf:.2f}")
axes[0].legend()
axes[0].grid(True)

# SBERT Plot
for label in df_sbert['label'].unique():
    subset = df_sbert[df_sbert['label'] == label]
    axes[1].scatter(subset['x'], subset['y'], label=f"Cluster {label}")
for _, row in df_sbert.iterrows():
    axes[1].text(row['x'] + 0.01, row['y'] + 0.01, row['text'], fontsize=8)
axes[1].set_title(f"SBERT + KMeans\nSilhouette: {silhouette_sbert:.2f}, CH Index: {calinski_sbert:.2f}")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
