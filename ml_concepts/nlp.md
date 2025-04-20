## 🗣️ Natural Language Processing (NLP)

NLP concepts, math, and practical examples, aiming to make everything intuitive.

---

### 🧠 1. Foundations of NLP

#### **Text Preprocessing**

Before doing any NLP task, we need to clean and standardize the text.

- **Tokenization**: Splits a sentence into words or subwords. For example, "NLP is cool!" becomes `["NLP", "is", "cool", "!"]`.
- **Stemming vs Lemmatization**:
  - *Stemming* cuts off word endings ("running" → "run") crudely.
  - *Lemmatization* finds the proper base form ("better" → "good").
- **Stopwords**: Words like "is", "the", "and" appear often but don’t add much meaning.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

text = "Natural Language Processing is amazing."
tokens = word_tokenize(text)
filtered = [w for w in tokens if w.lower() not in stopwords.words('english')]
print(filtered)  # ['Natural', 'Language', 'Processing', 'amazing', '.']
```

#### **Statistical Language Modeling**

A language model assigns a probability to a sequence of words. For example, a bigram model approximates:

$$ P(w_1, w_2, ..., w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-1}) $$

This says: "The probability of a sentence is the product of probabilities of each word given the one before it."

Smoothing (like Laplace smoothing) is used to handle zero probabilities when a bigram never appeared in training data.

#### **POS Tagging & Parsing**

- **POS Tagging**: Assigns grammatical roles like noun (NN), verb (VB).
- **Parsing**: Determines sentence structure (who did what to whom).

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a startup in the UK")
for token in doc:
    print(token.text, token.pos_)
```

---

### 📚 2. Topic Modeling & Text Classification

#### **TF-IDF (Term Frequency-Inverse Document Frequency)**

Used to find important words in documents:

$$ tf-idf(t, d) = tf(t, d) \cdot \log\left(\frac{N}{df(t)}\right) $$

Where:
- $tf(t, d)$ = term frequency of word *t* in document *d*
- $df(t)$ = number of documents containing word *t*
- $N$ = total number of documents

#### **LDA (Latent Dirichlet Allocation)**

LDA is a probabilistic model that assumes each document is a mix of topics, and each topic is a mix of words. For example, a news article might be 70% about politics and 30% about technology.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

corpus = ["Cats are small animals.", "Dogs are loyal pets."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
lda = LatentDirichletAllocation(n_components=2)
lda.fit(X)
```

#### **Text Classification**

We want to predict a label (like spam/not spam) from a sentence.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

corpus = ["Buy cheap meds now", "Meeting today at 3pm"]
labels = [1, 0]  # 1 = spam, 0 = not spam
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(corpus, labels)
```

---

### 🧬 3. Word Representations

#### **Word2Vec**

Instead of using words as symbols, we represent them as vectors that encode meaning. Word2Vec trains a neural network to predict context words (Skip-gram) or center word (CBOW).

```python
from gensim.models import Word2Vec
sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
print(model.wv['NLP'])
```

#### **GloVe**

GloVe learns word vectors by looking at how often words co-occur in a large corpus. Its cost function is:
$$J = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
Where $X_{ij}$ is how often word $i$ appears near word $j$.

#### **Contextual Embeddings (e.g., BERT)**

BERT gives different embeddings for the same word depending on its context.

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Bank of a river", return_tensors="pt")
outputs = model(**inputs)
```

---

### 🤖 4. Deep Learning for NLP

#### **RNN/LSTM/GRU**

These models handle sequences by maintaining a hidden state. They work well for small data but struggle with long dependencies.

$$ h_t = f(Wx_t + Uh_{t-1} + b) $$

#### **Attention Mechanism**

Instead of only using the last hidden state, attention looks at all words and weighs them based on relevance:

$$ Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

#### **Transformers**

Transformers replaced RNNs in most tasks. They use multi-head self-attention and feed-forward layers.

---

### 💬 5. Core NLP Tasks

Here are the typical NLP problems you’ll face:

- **Text Classification**: Predict category (e.g., spam, sentiment).
- **NER**: Find and label entities (e.g., "Apple" as ORG).
- **Question Answering**: Extract answers from a context paragraph.
- **Summarization**: Reduce long documents to key points.
- **Translation**: Translate from one language to another.
- **Text Generation**: Predict next words or complete a sentence.

```python
from transformers import pipeline
summarizer = pipeline("summarization")
print(summarizer("Text summarization is the process of..."))
```

---

### 🧪 6. Evaluation Metrics

#### **Classification Metrics**
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
- **F1 Score** = Harmonic mean of precision and recall  

#### **Generation Metrics**
- **BLEU**: Compares overlap with reference (used in translation).  
- **ROUGE**: Overlap of n-grams (used in summarization).  
- **Perplexity**: Measures how well a language model predicts a sample. Lower perplexity indicates better predictive performance. Mathematically, it's the exponential of the average negative log-likelihood:

$$\text{Perplexity} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{i-1}) \right)$$

```python
from sklearn.metrics import classification_report
print(classification_report([0, 1, 1], [0, 1, 0]))
```

---

### ⚙️ 7. Tooling & Libraries

- `transformers`: Hugging Face's library for BERT, GPT, etc.
- `spaCy`: Fast, easy-to-use NLP tools for production.
- `nltk`: Classic library for linguistic tasks.
- `gensim`: Topic modeling and Word2Vec.
- `scikit-learn`: Text classification and evaluation.

---

### 🧑‍🔬 8. Trends in NLP (2025)

- **LLMs in Production**: Focus on deployment, speed, and safety.
- **Retrieval-Augmented Generation (RAG)**: Use knowledge bases to boost LLMs.
- **Prompt Engineering**: Design smart inputs for better outputs.
- **Multilingual NLP**: Models trained to work across 100+ languages.
- **Edge NLP**: Light models for phones and IoT devices.

---

### 📘 9. NLP Sequence labeling

**Sentence:**  
👉 `"John gave Mary a book in London on Tuesday."`

| Task Type | Focus | Output |
|-----------|-------|--------|
| **POS (Part-of-Speech) tagging** | **Syntax:** What is each word grammatically? | `NNP VBD NNP DT NN IN NNP IN NNP` |
| **NER (Named Entity Recognition)** | **Semantics:** Is the word part of a real-world entity? | `John: PERSON, Mary: PERSON, London: LOC, Tuesday: DATE` |
| **Chunking (Shallow Parsing)** | **Groups:** Which words form a meaningful unit? | `[(John), (gave), (Mary), (a book), (in London), (on Tuesday)]` |
| **Dependency Parsing** | **Structure:** How do words relate grammatically? | `John → gave (subject), Mary → gave (indirect object), book → gave (direct object)` |
| **SRL (Semantic Role Labeling)** | **Roles:** Who is doing what to whom? | `Agent: John, Action: gave, Recipient: Mary, Theme: book, Location: London, Time: Tuesday` |
| **Coreference** | **Tracking:** Which mentions refer to the same thing? | `"John gave Mary a book. He..."` → `"John"` and `"He"` are linked |
