# 🧠 **Machine Learning Concepts Overview**

---

## 📚 Deep Learning & Neural Networks

### 1. Depth-wise Separable Convolution  
Efficient form of convolution used in lightweight architectures (e.g., MobileNet). Separates spatial and depth-wise operations to reduce parameters and computation.

### 2. Transformers & Attention Mechanisms  
- Transformers use self-attention to process sequences in parallel.  
- **Attention** scores each token's relevance within a sequence.  
- Key components:
  - Scaled Dot-Product Attention  
  - Multi-Head Attention  
  - Positional Encoding  
  - Encoder-Decoder structure (in vanilla transformers)

### 3. Recurrent Neural Networks (RNNs)  
- Sequential models maintaining memory via hidden states.  
- Limitations: Vanishing gradients, limited parallelization.  
- Variants: LSTM, GRU  
- Applications: Language modeling, time series, speech recognition.

### 4. U-Net Architecture  
- U-shaped architecture for image segmentation.  
- Combines contracting path (downsampling) with expansive path (upsampling) and skip connections for preserving spatial info.

### 5. Dropout  
- Regularization to prevent overfitting.  
- Randomly disables neurons during training to improve generalization.

### 6. Gradient Descent & Training Techniques  
- **Gradient Descent**: Core optimization technique for minimizing loss.  
- **Batch Normalization**: Stabilizes and accelerates training by normalizing activations.  
- **Training Parallelism**: Split training across GPUs (data/model/pipe parallelism).

---

## 📉 Loss Functions

### 1. Cross-Entropy Loss  
- Used in classification.  
- Measures dissimilarity between predicted and actual class probabilities.

### 2. Contrastive Loss  
- Used in Siamese networks and metric learning.  
- Encourages similar pairs to be close and dissimilar pairs far apart.

### 3. KL Divergence vs Cross-Entropy  
- KL Divergence measures how one probability distribution diverges from another.  
- Cross-Entropy includes KL divergence:  
  \[ H(p, q) = H(p) + KL(p || q) \]

### 4. Logistic Regression Loss  
- Log-loss (binary cross-entropy):  
  \[ \mathcal{L} = - \sum y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \]  
- Convex, enables global minimum optimization.

### 5. Multi-label Loss Functions  
- Use Binary Cross-Entropy + Sigmoid per label.  
- Each class prediction is independent.

---

## 📊 Model Evaluation & Metrics

### 1. Precision, Recall, F1-Score  
- **Precision**: TP / (TP + FP)  
- **Recall**: TP / (TP + FN)  
- **F1**: Harmonic mean of precision and recall.

### 2. RMSE vs MAE  
- **MAE**: Linear error metric, robust to outliers.  
- **RMSE**: Penalizes larger errors more (quadratic).

### 3. P-value  
- Probability of obtaining data as extreme as observed under the null hypothesis.

### 4. Classification Metrics  
- Common: Accuracy, Precision, Recall, F1, ROC-AUC.  
- Imbalanced datasets: Prefer PR-AUC and F1.

### 5. Threshold Selection Criteria  
- Use ROC curve or Precision-Recall curve.  
- Cost-sensitive thresholds.  
- Maximize metrics like F1 or minimize false positives.

---

## 🔄 Model Design & Optimization

### 1. Cross-Validation  
- Prevents overfitting, supports hyperparameter tuning.  
- Types: K-Fold, Stratified, Time Series Split, LOOCV.

### 2. Regularization  
- Prevents overfitting by penalizing large weights.  
- L1 (Lasso): Sparsity and feature selection.  
- L2 (Ridge): Weight shrinkage.

### 3. Feature Selection  
- **Filter**: Based on statistics (correlation, chi-square).  
- **Wrapper**: Uses model performance (RFE).  
- **Embedded**: Built-in (Lasso, Trees).

### 4. Feature Importance Techniques  
- Tree-based (Gini/entropy), permutation-based, SHAP, LIME.

### 5. Dimensionality Reduction  
- Linear: PCA, ICA  
- Nonlinear: t-SNE, UMAP  
- Supervised: LDA  
- Neural: Autoencoders

### 6. Curse of Dimensionality  
- Sparsity in high dimensions reduces effectiveness of distance metrics.  
- Solution: Dimensionality reduction, feature engineering.

### 7. Collinearity  
- High correlation between features distorts model coefficients.  
- Detect via correlation matrix or VIF (Variance Inflation Factor).

### 8. Error Propagation in Time Series  
- Recursive models accumulate error over time.  
- Use seq2seq or scheduled sampling to reduce compounding.

### 9. Overfitting Detection & Mitigation  
- Gap between training and validation metrics.  
- Techniques: Dropout, Regularization, Early Stopping, Data Augmentation.

### 10. Imbalanced Datasets  
- Oversample, undersample, SMOTE, class weighting, anomaly framing.  
- Evaluation via F1, PR-AUC over accuracy.

---

## 🤖 Core ML Algorithms

### 1. K-Nearest Neighbors (KNN)  
- Non-parametric, lazy learning.  
- Sensitive to feature scale and irrelevant features.

### 2. XGBoost vs Random Forest  
- **Random Forest**: Bagging ensemble.  
- **XGBoost**: Gradient boosting. More accurate but computationally intensive.

### 3. Bagging vs Boosting  
- **Bagging**: Reduces variance.  
- **Boosting**: Reduces bias. Models learn sequentially.

### 4. Tree Splitting Criteria  
- Gini Impurity  
- Information Gain (Entropy)  
- Gain Ratio

### 5. Ensemble Methods  
- Bagging (e.g., RF), Boosting (e.g., XGBoost), Stacking (meta-models).

### 6. Multi-Armed Bandits  
- Explore-exploit tradeoff.  
- MLE: Maximize likelihood.  
- Bayesian: Posterior distribution using priors.

### 7. Bayesian vs Frequentist  
- **Bayesian**: Parameters are distributions.  
- **Frequentist**: Parameters are fixed; uncertainty is from data.

### 8. EM Algorithm  
- Used with incomplete data or latent variables.  
- Alternates between estimating expected values (E-step) and maximizing parameters (M-step).

### 9. Dynamic Programming  
- Solves subproblems and stores results.  
- Used in DP algorithms like edit distance, knapsack, etc.

### 10. t-SNE  
- Preserves local structure in lower dimensions.  
- Optimizes KL divergence between high/low-dimensional pairs.

---

## 💻 System Design & Databases

### 1. Database Structure  
- Relational DBs (e.g., MySQL).  
- Concepts: Joins, Indexing, Normal Forms (1NF, 2NF, 3NF).

### 2. Recommendation System Design  
- Methods:  
  - Content-based  
  - Collaborative filtering  
  - Hybrid systems  
- Challenges: Cold-start, sparsity, implicit feedback.

### 3. Hierarchical Classification  
- Predict top-level classes, then sub-classes.  
- Flat vs Hierarchical models (big-bang vs local classifier chains).

### 4. Deterministic vs Non-Deterministic Models  
- Deterministic: Always same output.  
- Non-Deterministic: Includes randomness (e.g., Dropout in inference mode).

### 5. Context-Aware Entity Ranking  
- Combines user, item, and context embeddings.  
- Enhances relevance with time, location, or session features.

### 6. Query Embedding Design  
- Encode text using models like BERT, Word2Vec.  
- Aggregate via mean pooling, max pooling, or RNNs.  
- Fine-tune using relevance signals.

### 7. Model Auditing for Skewed Data  
- Analyze class imbalance and misclassification.  
- Metrics: Confusion matrix, PR curves, F1 for minority.

---

## 👨‍💻 Coding Problems

### 1. Grid Maze (Unique Paths II)  
- Use DP to count paths avoiding obstacles.

### 2. Modified Tree Traversal  
- Level order, zigzag, and reverse level traversal variations.

### 3. Balanced Parentheses  
- Use a stack to track matching brackets.

### 4. Dynamic Programming Basics  
- Identify and solve overlapping subproblems. Memoize results.

### 5. Binary Tree Inorder Traversal  
🔗 [Leetcode Editorial](https://leetcode.com/problems/binary-tree-inorder-traversal/editorial/)

### 6. Find Unique Element in Sorted Array  
- Use XOR or binary search based on index patterns.

### 7. Number of Pairs with Sum < K  
- Two-pointer after sorting or hash map for lookup.

### 8. Peak Load Interval from Login/Logout Times  
- Sweep line or priority queue to count active users.

### 9. Peak Load from Login Events Only  
- Sort events, simulate timeline to count users.

### 10. Sparse Vector Dot Product  
- Use dictionaries or compressed representation.  
- Optimize for sparsity.

### 11. Unique Customers Visiting Every Day  
- Use dictionary of sets per day, intersect all sets.
