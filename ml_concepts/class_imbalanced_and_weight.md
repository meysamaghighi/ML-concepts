# Libraries dealing with class imbalanced

When dealing with **imbalanced datasets**, there are several excellent Python libraries designed to help with resampling, evaluation, and specialized modeling techniques.

---

### 📦 **Top Libraries for Imbalanced Datasets**

| **Library** | **Purpose** | **Key Functions / Features** |
|-------------|-------------|-------------------------------|
| **[imbalanced-learn](https://imbalanced-learn.org/)** | Resampling strategies (oversampling, undersampling) | `SMOTE`, `RandomOverSampler`, `ADASYN`, `TomekLinks`, `ClusterCentroids` |
| **scikit-learn** | Modeling and evaluation (some built-in support for imbalance) | `class_weight='balanced'`, `roc_auc_score`, `precision_recall_curve` |
| **xgboost / lightgbm** | Gradient boosting with native imbalance handling | `scale_pos_weight` parameter |
| **CatBoost** | Gradient boosting with good class imbalance performance by default | No need for manual balancing in many cases |
| **TensorFlow / PyTorch** | Deep learning frameworks with support for weighted loss | `pos_weight` in loss functions like `BCEWithLogitsLoss` |
| **Yellowbrick** | Visualization tools for imbalanced classification | `class_balance`, `confusion_matrix`, `precision_recall_curve` visualizations |
| **skorch** | Wraps PyTorch models for sklearn compatibility | Enables integration of PyTorch + imbalanced-learn pipelines |

---

### 🎯 Common Techniques These Libraries Support

- **Oversampling** the minority class:
  - `SMOTE`, `ADASYN`, `BorderlineSMOTE`
- **Undersampling** the majority class:
  - `RandomUnderSampler`, `TomekLinks`
- **Cost-sensitive learning**:
  - Class weights or loss weighting (e.g., `class_weight='balanced'` in scikit-learn models)
- **Ensemble methods** tailored for imbalance:
  - `BalancedRandomForestClassifier`, `EasyEnsembleClassifier` (from imbalanced-learn)




# Class Weight in Scikit-Learn

In **scikit-learn**, several models accept a `class_weight` parameter, which allows the model to **treat classes differently during training**, especially useful for **imbalanced classification problems**.

---

### ✅ **What does `class_weight` do?**

When set, `class_weight` **adjusts the importance of each class** during training:
- **Minority classes** get **higher weights**, so errors on them are penalized more.
- **Majority classes** get **lower weights**, so the model doesn't overly favor them.

This impacts:
- **The loss function**: It’s modified to give more/less penalty to mistakes depending on class.
- **Decision thresholds**: Some models may shift the boundary toward the majority class unless weights are adjusted.

---

### 🧠 **How does it affect the learning?**
- For **linear models**, it changes the optimization objective (e.g., weighted logistic loss).
- For **tree-based models**, it affects how splits are evaluated (weighted Gini or entropy).
- For **SVMs**, it modifies the penalty parameter `C` per class.

---

### 🔧 **scikit-learn models that support `class_weight`**

| Model Type | Model Name | Accepts `class_weight` |
|------------|-------------|------------------------|
| Linear Models | `LogisticRegression` | ✅ |
| | `RidgeClassifier`, `Perceptron`, `SGDClassifier` | ✅ |
| Tree-Based | `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` | ✅ |
| | `GradientBoostingClassifier` | ❌ (does **not** natively support `class_weight`) |
| | `HistGradientBoostingClassifier` | ✅ |
| SVMs | `SVC`, `LinearSVC`, `NuSVC` | ✅ |
| Naive Bayes | ❌ | ❌ (but you can weight samples manually) |
| Ensemble | `BaggingClassifier` | ❌ (but you can pass weighted base estimators manually) |
| Neural Nets | `MLPClassifier` | ✅ |

---

### 🔁 **How to use it**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

Or define your own weights:
```python
model = LogisticRegression(class_weight={0: 1, 1: 3})
```

---

### 🎯 When to use?
- When your dataset is **imbalanced** (e.g., 95% vs 5% classes).
- Instead of **resampling**, or in combination with it.
- When false negatives/positives have **different costs**.

---
