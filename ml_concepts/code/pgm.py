# Demonstrating top use cases of PGMs with Python examples

# 1. Classification with Bayesian Network
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure
bn_model = DiscreteBayesianNetwork([('Disease', 'Fever'), ('Disease', 'Cough')])

# Define the CPDs
cpd_disease = TabularCPD(variable='Disease', variable_card=2, values=[[0.6], [0.4]])
cpd_fever = TabularCPD(variable='Fever', variable_card=2, 
                       values=[[0.9, 0.2], [0.1, 0.8]], evidence=['Disease'], evidence_card=[2])
cpd_cough = TabularCPD(variable='Cough', variable_card=2, 
                       values=[[0.7, 0.3], [0.3, 0.7]], evidence=['Disease'], evidence_card=[2])

# Add CPDs to the model
bn_model.add_cpds(cpd_disease, cpd_fever, cpd_cough)

# Inference
inference_bn = VariableElimination(bn_model)
classification_result = inference_bn.query(['Disease'], evidence={'Fever': 1, 'Cough': 1})

print("================================================")
print(classification_result)

# 2. Topic Modeling with LDA
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# Sample documents
docs = [["apple", "banana", "apple"], ["car", "bike", "car"], ["apple", "car", "banana"]]
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Train LDA model
lda_model = LdaModel(corpus=corpus, num_topics=2, id2word=dictionary, random_state=42)

# Get topics
lda_topics = lda_model.print_topics()

print("================================================")
for topic in lda_topics:
    print(topic)
    

# 3. Sequence Modeling with HMM
# from hmmlearn import hmm
# import numpy as np

# # Observation encoding
# obs_map = {'walk': 0, 'shop': 1, 'clean': 2}
# obs_sequence = np.array([[obs_map[o]] for o in ['walk', 'shop', 'clean']])

# # Define HMM with 2 hidden states
# model = hmm.MultinomialHMM(n_components=2, n_trials=None, n_iter=100, random_state=42)

# # Manually set model parameters
# model.startprob_ = np.array([0.6, 0.4])  # P(Start in Sunny, Rainy)
# model.transmat_ = np.array([
#     [0.7, 0.3],  # Sunny to Sunny, Rainy
#     [0.4, 0.6]   # Rainy to Sunny, Rainy
# ])
# model.emissionprob_ = np.array([
#     [0.6, 0.3, 0.1],  # P(walk, shop, clean | Sunny)
#     [0.1, 0.4, 0.5]   # P(walk, shop, clean | Rainy)
# ])

# # Run Viterbi decoding
# logprob, hidden_states = model.decode(obs_sequence, algorithm="viterbi")

# # Human-readable state names
# state_map = {0: 'Sunny', 1: 'Rainy'}
# decoded_states = [state_map[s] for s in hidden_states]
# print("Most likely hidden states:", decoded_states)

# 4. Approximate Inference with Gibbs Sampling
import random

def gibbs_sampler(num_iter=10):
    x, y = 0, 0
    samples = []
    for _ in range(num_iter):
        x = random.gauss(y, 1)  # P(x|y)
        y = random.gauss(x, 1)  # P(y|x)
        samples.append((x, y))
    return samples

gibbs_samples = gibbs_sampler(10)

print("================================================")
print(gibbs_samples)

# Collect results for display
import pandas as pd

results = {
    "Bayesian Network Inference": classification_result,
    "LDA Topics": lda_topics,
    # "HMM State Sequence": decoded_states,
    "Gibbs Samples": pd.DataFrame(gibbs_samples, columns=["x", "y"]).round(2)
}

# import ace_tools as tools; tools.display_dataframe_to_user(name="Gibbs Samples", dataframe=results["Gibbs Samples"])
# results.pop("Gibbs Samples")  # Already displayed

results
