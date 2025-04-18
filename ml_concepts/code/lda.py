import os
import re
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Path to repo root relative to this script's location
# repo_root = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".."))
repo_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Debug info
print(f"Searching for .md files in: {repo_root}")
md_files = []

# Collect tokenized documents from .md files
docs = []
for dirpath, _, filenames in os.walk(repo_root):
    for file in filenames:
        if file.endswith(".md"):
            file_path = os.path.join(dirpath, file)
            md_files.append(file_path)
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                tokens = word_tokenize(content.lower())
                tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
                if tokens:
                    docs.append(tokens)

# Show files processed
print("\nFiles included:")
for file_path in md_files:
    print(f" - {file_path}")

# Ensure there are documents to process
if not docs:
    raise ValueError("No valid content found in .md files after tokenization.")

# Create dictionary and corpus
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Train LDA model
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, random_state=42)

# Print topics
print("\n================================================")
print("LDA Topics:")
for topic in lda_model.print_topics():
    print(topic)
