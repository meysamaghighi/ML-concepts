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

Here's an updated and more detailed version of your text, with deeper explanations of **Gini impurity**, **entropy**, what gets maximized, and how the **feature to split on** is selected:

---

### 4. How to Split a Tree?

**Problem:** At each decision tree node, we must choose the best way to split the data to improve prediction quality.  
**Goal:** Choose splits that **reduce impurity** and **maximize information gain**, making the resulting child nodes more "pure" (i.e., dominated by a single class).

---

### ✅ **Key Metrics for Splitting**

#### 🔸 **Gini Impurity** (used in CART):
```math
\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2
```
- $p_i$ is the proportion of samples belonging to class $i$ in the node.
- Gini measures how *often a randomly chosen element would be incorrectly labeled* if it was randomly labeled according to the distribution in the node.
- **Lower Gini = purer node**.

#### 🔸 **Entropy** (used in ID3 and C4.5):
```math
\text{Entropy} = -\sum_{i=1}^{C} p_i \log_2(p_i)
```
- Entropy measures the **uncertainty** or **disorder** in the class distribution.
- Higher entropy = more mixed classes; Lower entropy = purer.
- A node where all samples belong to one class has **entropy = 0**.

---

### 🔍 What is *really* maximized across branches?

We **maximize the **Information Gain**, which is the **reduction in impurity** after a split.

#### Information Gain (IG):
```math
\text{IG} = \text{Impurity (parent)} - \sum_{k} \left( \frac{n_k}{n} \cdot \text{Impurity (child } k) \right)
```

Where:
- $n_k$ = number of samples in child $k$,
- $n$ = total samples in the parent,
- The impurity can be either **Gini** or **Entropy** depending on the algorithm.

So, we **maximize the impurity reduction** (or equivalently, **minimize the weighted impurity** of child nodes).

---

### 🌿 How is the best feature to split on selected?

At each node:
1. For **every feature**:
   - Try all reasonable **thresholds** (for numeric) or **groups** (for categorical).
   - Simulate a split using that feature and threshold.
   - Calculate the **information gain** (or Gini reduction).
2. Select the **feature and threshold** combination that gives the **highest information gain**.
3. Apply the split and repeat the process **recursively** for each child node.

---

### 🛑 Stop when:
- Max depth is reached.
- Node has too few samples (e.g., `min_samples_split`).
- Node is already pure (all samples same class).
- Information gain is below a threshold.

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
  ```math
  \text{Loss} = -\sum_{i=1}^L \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
  ```
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
    ```math
    J = TPR - FPR
    ```

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

