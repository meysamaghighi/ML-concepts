## 📈 Time Series & Distributions

**1. What is time series windowing and what are its benefits?**  
Slices a series into windows to help models learn short-term dependencies. Useful for supervised time series prediction.

**2. What are the main differences between ARIMA and SARIMA?**  
- **ARIMA:** Non-seasonal, uses AR, I, MA.  
- **SARIMA:** Adds seasonal AR, MA, and differencing.

**3. How can you incorporate features into time series models?**  
Use exogenous variables in models (e.g., ARIMAX, LSTM) or as lag-based features.

**4. How does time series decomposition work?**  
Splits data into **trend**, **seasonality**, and **residual** components for better modeling or analysis.

**5. What are the differences between confidence intervals and forecast intervals?**  
- **CI:** Uncertainty in estimated parameters.  
- **FI:** CI + noise in data → uncertainty in actual predictions.

**6. What metrics can you use to evaluate a time series model?**  
- Accuracy: **MAE, RMSE, MAPE, SMAPE**  
- Explained variance: **R²**  
- Interval quality: **Coverage, interval width**

**7. What metrics would you use to compare two distributions?**  
**KL divergence**, **Wasserstein distance**, **JS divergence**, **Hellinger distance**

---

## 📊 Statistics & Inference

**8. How does an A/B test work?**  
Splits users randomly to compare outcomes using hypothesis testing (e.g., t-test).

**9. What is shadow testing?**  
Deploy a model silently in production to compare its output with the live model.

**10. How do you build a confidence interval?**  
CI = estimate ± critical_value × std_error. Use normal assumptions or bootstrap.

**11. What is the difference between frequentist and Bayesian inference?**  
- **Frequentist:** Parameters fixed, uses likelihood.  
- **Bayesian:** Parameters as distributions, uses priors → posteriors via Bayes’ theorem.

**12. What is Bayes’ Theorem and how is it applied in ML?**  
$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$  
Used in Naive Bayes, Bayesian networks, optimization.

**44. MLE vs Bayesian inference?**  
- **MLE:** Finds parameters maximizing likelihood (point estimate).  
- **Bayes:** Uses priors and returns a distribution over parameters.

**45. MLE for regression – how can we do it?**  
Assuming Gaussian noise, minimizing squared error = maximizing log-likelihood.

**46. What is a p-value?**  
Probability of seeing the observed data if the null hypothesis were true.

**47. Importance of randomization in A/B tests?**  
Eliminates confounding variables, ensures fair comparison.

---

## 🧠 Core ML Concepts

**13. What is logistic regression and how does it work?**  
Applies a sigmoid to linear input to estimate class probabilities.

**14. How to choose the optimal decision threshold?**  
Use ROC or PR curves, depending on the cost of FP/FN and data balance.

**15. How can unbalanced datasets be handled?**  
- **Resampling**: Oversample minority / undersample majority  
- **Class weighting**  
- **Robust metrics**: F1, AUC

**16. How can missing data be handled?**  
- **Deletion**, **Imputation** (mean, KNN, regression), or  
- Model the missingness as a feature.

**17. What techniques can be used to handle categorical features?**  
**One-hot**, **label encoding**, **target encoding**, or **embeddings**

**18. What are feature embeddings and how are they used?**  
Low-dimensional dense vectors used in neural models for structured or categorical input.

**27. What are the main regularization techniques?**  
- **L1 (Lasso):** Feature selection  
- **L2 (Ridge):** Weight shrinkage  
- **Dropout**, **BatchNorm**: Neural net regularization

**26. How can bias and variance be traded off?**  
- Use **simpler models**, **regularization**, **ensembles**, and **cross-validation**.

**53. What is overfitting? How can it be identified and mitigated?**  
Occurs when model memorizes data. Mitigate via regularization, simpler models, early stopping.

---

## 🔍 Evaluation & Generalization

**25. What are the main evaluation metrics for classification and regression?**  
- **Classification:** Accuracy, F1, Precision, Recall, AUC  
- **Regression:** MSE, RMSE, MAE, R²

**48. What is cross-validation and why is it important?**  
Technique to estimate model generalization. Variants: **k-fold**, **LOO**, **Stratified**.

**49. What is the Curse of Dimensionality?**  
Distance metrics degrade in high-dimensional space. Hurts nearest-neighbor-based models.

**43. What is the Hoeffding Inequality?**  
Gives probability bounds for deviation from expected value. Used in generalization analysis.

---

## 🧠 Deep Learning & Representation Learning

**19. What is the principle behind dimensionality reduction?**  
Project high-dimensional data into fewer dimensions.  
- Linear: **PCA**  
- Non-linear: **t-SNE**, **UMAP**  
- Neural: **Autoencoders**, **VAEs**

**20. What is boosting and bagging? How do they differ?**  
- **Bagging:** Independent models (e.g., RF), reduces variance  
- **Boosting:** Sequentially built, corrects errors, reduces bias (e.g., XGBoost)

**21. How does ensemble learning work?**  
Combine models using **bagging**, **boosting**, or **stacking** to improve performance.

**28. How does a Bayesian generative model like a VAE work?**  
- Encodes to a distribution  
- Samples and decodes  
- Loss = reconstruction + KL divergence

---

## 🤖 Neural Architectures & XAI

**22. CNNs, GNNs, Transformers – how do they work?**  
- **CNNs:** Convolutions for local spatial features  
- **GNNs:** Aggregate graph node and edge info  
- **Transformers:** Attention over sequence, handles long dependencies

**23. What is attention and how is it used in XAI?**  
Assigns weights to parts of input. Visualized for interpretability (e.g., attention heatmaps).

**24. Transformers vs RNNs – pros and cons?**  
- **Transformers:** Parallel, handle long context well  
- **RNNs:** Lower memory, better on small data

**50. What is the time complexity of Transformers?**  
Self-attention: $O(n^2d)$, with $n$ = sequence length, $d$ = dimension.

**51. How does a Transformer work?**  
Uses self-attention, positional encoding, feedforward layers, residual connections.

**52. What happens if you remove all hidden layers in a neural net?**  
It becomes a linear model (i.e., logistic or linear regression).

**54. What are different types of CNNs?**  
Standard, Dilated, Depth-wise separable, Transposed

**55. What is a depth-wise separable convolution?**  
Split standard convolution into channel-wise + 1×1 point-wise. Efficient.

---

## 🧩 Learning Types, KNN, Bandits

**29. Supervised vs Unsupervised vs Semi-supervised Learning?**  
- **Supervised:** Labeled (e.g., regression)  
- **Unsupervised:** Unlabeled (e.g., clustering)  
- **Semi-supervised:** Few labels + many unlabeled

**30. Can you explain the k-nearest neighbors (KNN) algorithm?**  
Classifies based on the closest k samples (e.g., Euclidean distance).

**31. How to choose ‘k’ in KNN?**  
Cross-validation. Low k = overfitting, high k = underfitting.

**32. What are multi-armed bandits?**  
Explore-exploit strategy. Algorithms: ε-greedy, UCB, Thompson Sampling.

---

## 🧠 Generative AI, NLP, Causality

**33. How does a Bayesian generative model like VAE work?**  
(See #28)

**34. Describe a typical NLP pipeline.**  
Text cleaning → Tokenization → Embedding → Model → Output  
Optional: Lemmatization, POS tagging

---

## ⚙️ Optimization

**35. What is the difference between first- and second-order optimizers?**  
- **First-order:** Gradients (SGD, Adam)  
- **Second-order:** Hessians (Newton's), more accurate, slower.

**36. What is regularization and why is it used?**  
Adds penalty to loss function to avoid overfitting. See Q27.

**37. What is the time complexity of matrix inversion?**  
Typically $O(n^3)$, but optimized versions exist (Cholesky, LU).

**38. What is the closed-form solution of linear regression?**  
$\theta = (X^TX)^{-1}X^Ty$.  
Used for small/medium data. For large data, use gradient descent.

---

## 🧪 Software Engineering & Algorithms

**39. Divide a number without using *, /, or %?**  
Use subtraction and bit shifting to simulate division.

**40. Modified tree traversal question?**  
Depends on the variant (e.g., inorder without recursion, postorder with extra state).

**41. How would you handle a short deadline for a project?**  
Prioritize MVP, communicate early, reuse components, timebox, and stay impact-focused.

---

