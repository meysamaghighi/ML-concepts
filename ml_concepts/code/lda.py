import os
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Path to repo root relative to this script's location
repo_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Debug info
print(f"Searching for .md files in: {repo_root}")
md_files = []

# Collect tokenized documents from .md files
docs = []
file_doc_map = []  # maps filename to doc index
for dirpath, _, filenames in os.walk(repo_root):
    for file in filenames:
        if file.endswith(".md"):
            file_path = os.path.join(dirpath, file)
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                tokens = word_tokenize(content.lower())
                tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
                if tokens:
                    docs.append(tokens)
                    md_files.append(file_path)
                    file_doc_map.append((file_path, len(docs)-1))

# Show files processed
print("\nFiles included:")
for file_path in md_files:
    print(f" - {file_path}")

if not docs:
    raise ValueError("No valid content found in .md files after tokenization.")

# Create dictionary and corpus
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Train LDA model
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, random_state=42)

# Print overall topics
print("\n================================================")
print("🧠 LDA Topics (word distribution):")
for topic_id, topic_words in lda_model.print_topics():
    print(f"Topic {topic_id}: {topic_words}")

# Print per-document topic distributions
print("\n================================================")
print("📄 Document-wise topic distributions:")

doc_topic_matrix = []
for i, bow in enumerate(corpus):
    topic_dist = lda_model.get_document_topics(bow)
    doc_topic_matrix.append([prob for _, prob in topic_dist])
    
    # Most probable topic
    top_topic = max(topic_dist, key=lambda x: x[1])
    print(f"[Doc {i}] {os.path.basename(md_files[i])}")
    print(f"  Main Topic: {top_topic[0]} (weight={top_topic[1]:.2f})")
    print(f"  Full Distribution: {topic_dist}")

# Print topic keywords cleanly
print("\n================================================")
print("🔑 Top Keywords per Topic:")
for topic_id in range(lda_model.num_topics):
    words_probs = lda_model.show_topic(topic_id, topn=10)
    keywords = ", ".join([word for word, _ in words_probs])
    print(f"Topic {topic_id}: {keywords}")

# Optional: print matrix form
np.set_printoptions(precision=3, suppress=True)
print("\n================================================")
print("📊 Document-Topic Matrix (rows=docs, cols=topics):")
# Create a consistent doc-topic matrix: rows = docs, cols = topics
num_topics = lda_model.num_topics
doc_topic_matrix = []

for bow in corpus:
    topic_dist = lda_model.get_document_topics(bow)
    # Convert sparse distribution to full vector
    full_dist = [0.0] * num_topics
    for topic_id, prob in topic_dist:
        full_dist[topic_id] = prob
    doc_topic_matrix.append(full_dist)

# Convert to NumPy array for nice display
doc_topic_matrix_np = np.array(doc_topic_matrix)

# Pretty print matrix
np.set_printoptions(precision=3, suppress=True)
print("\n================================================")
print("📊 Document-Topic Matrix (rows=docs, cols=topics):")
print(doc_topic_matrix_np)

