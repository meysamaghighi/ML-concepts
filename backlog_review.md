# 🧠 Machine Learning Study Guide

---

## 📚 Supervised Learning Topics

**Linear Regression**: Predicts a continuous target using a linear relationship between input variables and the output.

**Logistic Regression**: A classification algorithm used for binary outcomes using the sigmoid function.

**Naive Bayes**: A probabilistic classifier based on Bayes' Theorem assuming feature independence.

**Bagging & Boosting**: Ensemble methods; bagging reduces variance (e.g., Random Forest), boosting reduces bias (e.g., AdaBoost, XGBoost).

**K-nearest Neighbors (KNN)**: Classifies based on the majority class among the k closest samples.

**Decision Trees**: Splits data based on feature values to form a tree structure for classification or regression.

**Neural Networks**: Composed of layers of interconnected nodes used for complex pattern recognition.

**Support Vector Machines (SVMs)**: Finds the hyperplane that best separates classes with maximum margin.

**Random Forests**: An ensemble of decision trees using bagging and feature randomness.

**Gradient Boosted Trees**: Trees are trained sequentially to correct errors of previous trees.

**Kernel Methods**: Implicitly maps data into higher dimensions using kernel functions.

**Stochastic Gradient Descent (SGD)**: An optimization algorithm that updates weights incrementally.

**Sequence Modeling**: Models data where the order matters, often using RNNs or Transformers.

**Bayesian Linear Regression**: Adds probabilistic perspective to linear regression with distributions over weights.

**Gaussian Processes**: A non-parametric Bayesian approach to regression and classification.

**Overfitting / Underfitting**: Overfitting: model too complex. Underfitting: model too simple.

**Regularization**: Techniques like L1/L2 penalty to prevent overfitting.

**Evaluation Metrics**: Accuracy, precision, recall, F1-score, R-squared, MAE, MSE.

---

## 🔍 Unsupervised Learning Topics

**Clustering Algorithms**: Group data based on similarity (e.g., k-means, DBSCAN).

**K-means Clustering**: Assigns data to k clusters minimizing within-cluster variance.

**Anomaly Detection**: Identifies data points that deviate significantly from the norm.

**Markov Methods**: Uses state transition probabilities for modeling sequences.

**DBSCAN**: Density-based clustering algorithm that finds arbitrarily shaped clusters.

**Self-organizing Maps (SOMs)**: Neural networks that reduce dimensions while preserving topological properties.

**Deep Belief Nets**: Stacked RBMs for learning hierarchical representations.

**Expectation Maximization (EM)**: Iterative method for finding parameters in models with latent variables.

**Gaussian Mixture Models (GMMs)**: Probabilistic model assuming data is generated from a mixture of Gaussians.

**Clustering Evaluation**: Silhouette score, Davies–Bouldin index, adjusted Rand index.

---

## 🔗 Probabilistic Graphical Models

**Bayesian Networks**: Directed graphs representing conditional dependencies via Bayes' rule.

**Markov Networks**: Undirected graphs representing joint distribution via cliques.

**Variational Inference**: Approximate inference by turning inference into optimization.

**Markov Chain**: A memoryless stochastic process.

**Monte Carlo Methods**: Approximate solutions using repeated random sampling.

**Latent Dirichlet Allocation (LDA)**: A topic modeling technique assuming documents are mixtures of topics.

**Belief Propagation**: Inference algorithm for PGMs using message passing.

**Gibbs Sampling**: MCMC algorithm sampling each variable conditionally.

---

## 🔽 Dimensionality Reduction

**Autoencoders**: Neural networks used to learn compressed representations.

**t-SNE**: Nonlinear technique for embedding high-dimensional data for visualization.

**PCA**: Projects data onto orthogonal directions of maximum variance.

**SVD**: Matrix factorization technique useful for noise reduction and latent structure.

**Spectral Clustering**: Uses eigenvectors of similarity matrix for clustering.

**Matrix Factorization**: Used in recommendation systems (e.g., SVD, NMF).

---

## ⏱️ Sequential Models

**Hidden Markov Models (HMMs)**: Models sequences with hidden states and observable outputs.

**Conditional Random Fields (CRFs)**: Predict sequences of labels considering context.

**RNNs**: Neural networks for sequential data with memory of previous states.

**Named Entity Recognition (NER)**: Identifying entities like people and locations in text.

**Part-of-Speech Tagging**: Assigning grammatical tags to words.

---

## 🎮 Reinforcement Learning

**SARSA**: On-policy RL method updating values using the action taken.

**Explore-Exploit**: Balancing exploration of new actions and exploitation of known rewards.

**Multi-Armed Bandits**: Simple RL problem balancing exploration vs exploitation.

**Q-learning**: Off-policy RL method learning the value of actions regardless of policy.

**DQN**: Combines Q-learning with deep neural networks.

---

## 🧠 Deep Neural Networks / Deep Learning

**Feed-forward NN**: Basic neural network with inputs flowing forward.

**CNN**: Uses convolution layers for image and spatial data.

**Backpropagation**: Algorithm for computing gradients in neural networks.

**RNNs / LSTMs**: Capture dependencies in sequences.

**GANs**: Generative models with a generator and discriminator.

**Attention**: Allows models to focus on relevant input parts (used in Transformers).

**Dropout**: Regularization technique to prevent overfitting.

**Vanishing Gradient**: Problem in deep networks where gradients shrink.

**Activation Functions**: Non-linearities like ReLU, Sigmoid, Tanh.

---

## 🗣️ NLP Topics

**Statistical Language Modeling**: Assigns probabilities to sequences of words.

**LDA**: Topic modeling assuming documents are mixtures of topics.

**NER**: See above.

**Word Embedding**: Dense vector representation of words.

**Word2Vec**: Predicts context or word from surroundings.

**Sentiment Analysis**: Classify text by emotional tone.

**BERT**: Transformer-based model using bidirectional context.

**ULMFiT**: Transfer learning for text using fine-tuned language models.

---

## 🖼️ Image & Computer Vision

**Object Detection**: Identifying objects in images with bounding boxes.

**Image Recognition**: Classifying the main object or scene in an image.

**Pattern Recognition**: Detect patterns and regularities in data.

**FaceNet**: Embedding for face recognition.

**CNN**: See above.

**YOLO**: Real-time object detection algorithm.

---

## ⚙️ Training and Optimization

**Adaptive Gradient**: Algorithms like Adam, RMSprop that adapt learning rates.

**Regularization**: See above.

**Loss Functions**: Measures model error; e.g., Cross-Entropy, MSE.

**Bayesian vs MLE**: Bayesian incorporates prior, MLE maximizes likelihood.

**Class Imbalance**: Techniques like resampling, synthetic data.

**K-Fold CV**: Splits data into k parts for robust evaluation.

**Bias vs Variance**: Trade-off between model complexity and generalization.

---

## 📏 Evaluation Metrics

**Accuracy**: (TP+TN)/Total

**Precision**: TP / (TP + FP)

**Recall**: TP / (TP + FN)

**ROC AUC**: Trade-off between TPR and FPR.

**R-squared**: Variance explained by regression model.

**MAP**: Mean of average precisions over queries.

**MRR**: Mean of reciprocal ranks of first relevant item.

**Equal Error Rate**: Where false acceptance = false rejection rate.

**A/B Testing**: Comparing two variants statistically.

---
