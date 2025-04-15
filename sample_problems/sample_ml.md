### **1. How to check overfitting? How to deal with it?**

**Problem**: Your model performs well on training data but poorly on validation/test data.  
**Goal**: Detect and mitigate overfitting to ensure generalization.

**Answer**:
- **Check Overfitting**:
  - Large gap between training and validation accuracy/loss.
  - Visual inspection of loss curves over epochs.

- **Deal with Overfitting**:
  - Regularization (L1/L2 penalties).
  - Early stopping during training.
  - Dropout (for neural networks).
  - Reduce model complexity (e.g., fewer layers or shallower trees).
  - Increase training data (data augmentation or collection).
  - Use cross-validation.

---

### **2. Could you design a query embedding for Amazon teams?**

**Problem**: Represent user queries in vector space for use in search, ranking, or recommendation systems.  
**Goal**: Capture semantic meaning and context of queries for downstream tasks.

**Answer**:
1. **Preprocessing**: Tokenize and normalize the query text.
2. **Embedding Layer**: Use pre-trained models (e.g., BERT, SBERT) or train custom embeddings.
3. **Contextualization**: Fine-tune embeddings using Amazon-specific signals like click-through data.
4. **Pooling**: Aggregate token embeddings via [CLS], mean pooling, or max pooling.
5. **Training Objective**: Contrastive loss, triplet loss, or supervised cross-entropy on relevant/irrelevant query-document pairs.
6. **Evaluation**: Use precision@k, recall@k, or NDCG on relevance tasks.

---

### **3. What is an ensemble algorithm (e.g., Random Forest)?**

**Problem**: Single models (like decision trees) often suffer from high variance or bias.  
**Goal**: Combine multiple models to improve stability and accuracy.

**Answer**:
- **Random Forest**:
  - Type: Bagging ensemble of decision trees.
  - Each tree: Trained on a **bootstrap sample** (random sampling with replacement).
  - Splits: Chosen from a **random subset of features** at each node.
  - Prediction: Majority vote (classification) or average (regression).
  - **Reduces variance** by averaging across many weak learners.

---

### **4. How to split a tree?**

**Problem**: At each decision tree node, we must choose the best way to split data to increase model purity.  
**Goal**: Split to maximize information gain and reduce impurity.

**Answer**:
- Use metrics like:
  - **Gini Impurity** (CART):  
    \[
    Gini = 1 - \sum p_i^2
    \]
  - **Information Gain / Entropy** (ID3):  
    \[
    Entropy = -\sum p_i \log p_i
    \]
- Choose feature and threshold that results in the best impurity reduction.
- Repeat recursively until stopping conditions (max depth, min samples, etc.).

---

### **5. What metrics would you use in a classification problem?**

**Problem**: You need to evaluate a classification model’s performance.  
**Goal**: Choose metrics that reflect the model’s usefulness and error types.

**Answer**:
- **Accuracy**: Good for balanced datasets.
- **Precision**: TP / (TP + FP) — for false positive-sensitive tasks.
- **Recall**: TP / (TP + FN) — for false negative-sensitive tasks.
- **F1-score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Trade-off between true positive and false positive rates.
- **PR-AUC**: Especially useful for imbalanced datasets.
- **Confusion matrix**: For a detailed breakdown of predictions.

---

### **6. How to deal with an imbalanced dataset?**

**Problem**: The model is biased toward the majority class due to class imbalance.  
**Goal**: Improve performance on the minority class while keeping overall performance stable.

**Answer**:
- **Resampling techniques**:
  - Oversample minority (e.g., SMOTE).
  - Undersample majority.
- **Class weights**: Assign higher cost to minority class in the loss function.
- **Use appropriate metrics**: Precision, recall, F1, PR-AUC instead of accuracy.
- **Try anomaly detection or specialized models** when imbalance is extreme.

---

### **7. What loss function do you use for multi-label problems?**

**Problem**: Each instance can belong to multiple labels (not mutually exclusive).  
**Goal**: Model should predict probabilities for each label independently.

**Answer**:
- Use **Binary Cross-Entropy Loss** applied independently to each label:
  \[
  \text{Loss} = -\sum_{i=1}^L \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
  \]
- If classes are imbalanced, use **weighted BCE** or **Focal Loss**.
- For evaluation: use **macro/micro F1-score**, **subset accuracy**, or **mean average precision (mAP)**.

---

### **8. How to choose threshold for classifier predicting Prime signup?**

**Problem**: Need to convert predicted probabilities into binary decisions.  
**Goal**: Choose threshold that balances business needs (e.g., cost vs gain).

**Answer**:
- Plot **Precision-Recall** and **ROC curves**.
- Choose threshold that:
  - Maximizes **F1-score** (balance of precision/recall).
  - Meets business constraints (e.g., at least 90% precision).
  - Optimizes a **custom profit/loss function**.
  - Uses **Youden’s J statistic**:  
    \[
    J = TPR - FPR
    \]

---

### **9. Reviewing a model with 1B positive and 200K negative samples**

**Problem**: Huge class imbalance; risk of overpredicting the dominant class.  
**Goal**: Ensure the model generalizes well and doesn’t ignore minority class.

**Answer**:
- **Check metrics beyond accuracy**: Use **PR-AUC**, **F1**, confusion matrix.
- **Model calibration**: Are predicted probabilities meaningful?
- **Performance on minority class**: Ensure recall and precision are acceptable.
- **Stratified validation**: To maintain class distribution during training/validation.
- **Threshold tuning**: May need different thresholds per class.
- **Bias detection**: Check if model always predicts the majority class.

---

### **10. How to train a context-aware entity ranking model?**

**Problem**: Rank entities (e.g., product names) based on query and surrounding context.  
**Goal**: Return the most relevant entities based on query intent and context.

**Answer**:
1. **Inputs**: User query + context (session history, location, time, past interactions).
2. **Model architecture**:
   - Use **transformers** or **dual-encoders** to embed both query and entities.
   - Incorporate context via concatenation, attention, or metadata embeddings.
3. **Training**:
   - Use **pairwise (e.g., hinge loss)** or **listwise (e.g., ListNet, softmax)** ranking loss.
   - Train on **click logs** or relevance-labeled data.
4. **Output**: Ranked list of entities with relevance scores.
5. **Evaluation**: Use **NDCG@k**, **MRR**, or **Precision@k**.

