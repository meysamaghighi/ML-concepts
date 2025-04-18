### 🔗 **Probabilistic Graphical Models (PGMs)**  
**Intuition**:  
PGMs are a way to represent complex joint probability distributions using graphs. Nodes represent random variables, and edges represent probabilistic dependencies. The main goal is to simplify understanding and computation by using the structure of the graph.

**Two main types**:
1. **Bayesian Networks** (directed)
2. **Markov Networks** (undirected)

**Use case**:  
Used in various areas like medical diagnosis, NLP, computer vision, recommendation systems, etc.

---

### ✅ **Bayesian Networks (BNs)**  
**Intuition**:  
Directed acyclic graphs (DAGs) where each node is a variable and edges imply direct probabilistic dependencies. Each node has a conditional probability distribution (CPD) given its parents.

**Use case**:  
Medical diagnosis (e.g., probability of disease given symptoms).

**Example code** (using `pgmpy`):
```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('Disease', 'Symptom')])
cpd_disease = TabularCPD('Disease', 2, [[0.8], [0.2]])
cpd_symptom = TabularCPD('Symptom', 2, [[0.9, 0.3], [0.1, 0.7]], evidence=['Disease'], evidence_card=[2])

model.add_cpds(cpd_disease, cpd_symptom)
inference = VariableElimination(model)
print(inference.query(['Disease'], evidence={'Symptom': 1}))
```

**Comparison**:
- More expressive than Naive Bayes.
- Better at handling causal reasoning than Markov networks.

---

### ✅ **Markov Networks (Markov Random Fields)**  
**Intuition**:  
Undirected graphical models. They encode dependencies using cliques and potential functions instead of conditional probabilities.

**Use case**:  
Image segmentation, where each pixel’s label depends on its neighbors.

**Example code** (conceptual — `pgmpy` has limited support):
```python
from pgmpy.models import MarkovNetwork

model = MarkovNetwork()
model.add_edges_from([('A', 'B'), ('B', 'C')])
# You would define potential functions instead of CPDs
```

**Comparison**:
- More suitable for symmetric relationships (e.g., spatial data).
- Unlike BNs, they don't model causality.

---

### ✅ **Variational Inference (VI)**  
**Intuition**:  
Approximate inference method. Instead of sampling, VI turns inference into an optimization problem by approximating the posterior with a simpler distribution.

**Use case**:  
Topic modeling, large-scale Bayesian inference.

**Example (with Pyro)**:
```python
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Define model and guide (approx. posterior)
def model(data):
    mu = pyro.sample("mu", dist.Normal(0, 1))
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(mu, 1), obs=data)

def guide(data):
    mu_loc = pyro.param("mu_loc", torch.tensor(0.))
    mu_scale = pyro.param("mu_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("mu", dist.Normal(mu_loc, mu_scale))

# Inference
svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
```

**Comparison**:
- Faster than MCMC in large datasets.
- May converge to local optima.

---

### ✅ **Markov Chain**  
**Intuition**:  
A stochastic process where the future state depends only on the current state (memoryless).

**Use case**:  
Weather modeling, board games (e.g., Monopoly), random walks.

**Example**:
```python
import numpy as np

P = np.array([[0.7, 0.3], [0.4, 0.6]])  # Transition matrix
state = 0
for _ in range(10):
    state = np.random.choice([0, 1], p=P[state])
    print(state)
```

**Comparison**:
- Basis for many algorithms like Gibbs Sampling, Hidden Markov Models, etc.

---

### ✅ **Monte Carlo Methods**  
**Intuition**:  
Use random sampling to estimate numerical results, especially for integration or expectations.

**Use case**:  
Estimating π, Bayesian inference, physics simulations.

**Example**:
```python
import random
inside = 0
for _ in range(10000):
    x, y = random.random(), random.random()
    if x**2 + y**2 <= 1:
        inside += 1
print("Estimated Pi:", (inside / 10000) * 4)
```

**Comparison**:
- More flexible but potentially slower than VI.
- Underlies methods like MCMC, Gibbs sampling.

---

### ✅ **Latent Dirichlet Allocation (LDA)**  
**Intuition**:  
A generative model where documents are mixtures of topics and topics are distributions over words.

**Use case**:  
Topic modeling in NLP.

**Example (with `gensim`)**:
```python
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

docs = [["cat", "dog", "mouse"], ["dog", "lion", "tiger"], ["mouse", "rat", "hamster"]]
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]
lda = LdaModel(corpus, num_topics=2, id2word=dictionary)
print(lda.print_topics())
```

**Comparison**:
- Similar to pLSA but with Dirichlet priors for generalization.
- Can be inferred using Gibbs Sampling or Variational Inference.

---

### ✅ **Belief Propagation (Message Passing)**  
**Intuition**:  
An inference algorithm where each node passes "messages" to neighbors until convergence. Works best on tree-like graphs.

**Use case**:  
Error correction in coding theory, image restoration.

**Example**:  
Hard to code from scratch, but libraries like `pgmpy` or `libDAI` implement it.

**Comparison**:
- Exact inference for trees.
- Approximate inference for loopy graphs (Loopy BP).

---

### ✅ **Gibbs Sampling**  
**Intuition**:  
A special case of MCMC where we sample one variable at a time conditioned on the rest.

**Use case**:  
Posterior estimation in Bayesian models, LDA inference.

**Example**:
```python
import random

def gibbs_sampler(num_iter):
    x, y = 0, 0
    for _ in range(num_iter):
        x = random.gauss(y, 1)  # Sample x given y
        y = random.gauss(x, 1)  # Sample y given x
        print(x, y)

gibbs_sampler(10)
```

**Comparison**:
- Easier to implement than Metropolis-Hastings.
- Can be slow to mix (slow convergence).

