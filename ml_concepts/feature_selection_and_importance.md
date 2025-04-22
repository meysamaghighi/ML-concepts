## 🔍 Feature Selection

Feature selection is the **process of choosing a subset of input features** that contribute the most to the predictive performance of the model. It's typically done **before** training the final model to:
- Improve model performance (speed, generalization)
- Reduce overfitting
- Improve interpretability

### 📂 Categories of Feature Selection Methods

#### 1. **Filter Methods**
These methods evaluate features independently of the model.

- **Techniques**:
  - **Correlation coefficients**: Remove highly correlated features (e.g., Pearson)
  - **Mutual Information (MI)**: Measures shared information between feature and target
  - **Chi-squared test**: For categorical features vs categorical labels
  - **Variance Threshold**: Removes features with low variance

```python
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.1).fit_transform(X)
```

---

#### 2. **Wrapper Methods**
These methods evaluate subsets of features using a predictive model.

- **Techniques**:
  - **Forward Selection**: Start with no features, add one at a time
  - **Backward Elimination**: Start with all features, remove one at a time
  - **Recursive Feature Elimination (RFE)**: Train model, rank features, recursively remove the least important

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(RandomForestClassifier(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

---

#### 3. **Embedded Methods**
Feature selection is performed as part of the model training.

- **Techniques**:
  - **Lasso (L1 regularization)**: Pushes coefficients of less useful features to zero
  - **Tree-based models**: Select based on built-in importance scores

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X, y)
```

---

## 💡 Feature Importance

Feature importance **quantifies how useful each feature was** in constructing a particular model.

### 📊 Model-Based Importance

#### 1. **Tree-Based Models** (e.g., Random Forest, XGBoost)
- Importance is usually based on how much each feature reduces impurity (e.g., Gini or entropy).
- Can be extracted using `.feature_importances_`.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X, y)
importances = model.feature_importances_
```

#### 2. **Permutation Importance**
- Shuffle each feature and measure how the performance drops.
- Captures true impact on model performance.

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X, y)
```

#### 3. **SHAP Values (SHapley Additive exPlanations)**
- A unified measure of feature importance, explaining **individual predictions**.
- Based on cooperative game theory.

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

#### 4. **Coefficient Magnitude** (Linear models)
- In linear/logistic regression, larger absolute values of coefficients imply more importance.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X, y)
importance = model.coef_
```

---

## 🧠 Summary Table

| Method                    | Model-Agnostic | Intuitive | Handles Interactions | Computes Importance |
|---------------------------|----------------|-----------|----------------------|----------------------|
| Correlation/Chi²          | ✅             | ✅        | ❌                   | ❌                   |
| RFE                      | ❌             | ✅        | ✅                   | ✅                   |
| Lasso                    | ❌             | ✅        | ❌                   | ✅                   |
| Random Forest Importances| ❌             | ✅        | ✅                   | ✅                   |
| Permutation Importance   | ✅             | ✅        | ✅                   | ✅                   |
| SHAP                     | ✅             | ✅✅✅    | ✅✅✅                | ✅✅✅                |

---

