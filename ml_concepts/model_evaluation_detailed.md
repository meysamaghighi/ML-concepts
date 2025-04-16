# Machine Learning Model Evaluation Methods

Analyzing and evaluating a machine learning (ML) model involves assessing its performance, robustness, fairness, and interpretability. Below is a comprehensive list of evaluation methods, categorized based on what they measure, with explanations, formulas, and sample Python code.

## 1. Performance Metrics
These measure how well a model makes predictions.

---
### Classification Models

- **Accuracy** <span style="color:#90EE90">(Best for balanced datasets)</span>
  - **Explanation**: Ratio of correct predictions to total predictions.
  - **Formula**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $
  - **Python**:
    ```python
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true, y_pred)
    ```

- **Precision** <span style="color:#FFD700">(Useful when false positives are costly)</span>
  - **Explanation**: Proportion of predicted positives that are actual positives.
  - **Formula**: $\text{Precision} = \frac{TP}{TP + FP} $
  - **Python**:
    ```python
    from sklearn.metrics import precision_score
    precision_score(y_true, y_pred)
    ```

- **Recall (Sensitivity)** <span style="color:#00CED1">(Useful when false negatives are costly)</span>
  - **Explanation**: Proportion of actual positives that were correctly identified.
  - **Formula**: $\text{Recall} = \frac{TP}{TP + FN} $
  - **Python**:
    ```python
    from sklearn.metrics import recall_score
    recall_score(y_true, y_pred)
    ```

- **F1 Score** <span style="color:#FF6347">(Useful when both false positives and false negatives matter)</span>
  - **Explanation**: Harmonic mean of precision and recall.
  - **Formula**: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $
  - **Python**:
    ```python
    from sklearn.metrics import f1_score
    f1_score(y_true, y_pred)
    ```

- **ROC-AUC** <span style="color:#DA70D6">(Useful for imbalanced datasets)</span>
  - **Explanation**: Measures the area under the ROC curve.
  - **Formula**: Area under TPR vs. FPR curve.
  - **Python**:
    ```python
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_true, y_scores)
    ```

- **PR-AUC** <span style="color:#FFA07A">(Useful when positive class is rare)</span>
  - **Explanation**: Area under the precision-recall curve.
  - **Python**:
    ```python
    from sklearn.metrics import average_precision_score
    average_precision_score(y_true, y_scores)
    ```

- **Log Loss (Cross-Entropy)** <span style="color:#7FFFD4">(Useful for probabilistic models)</span>
  - **Explanation**: Measures accuracy of predicted probabilities.
  - **Formula**: $-\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)] $
  - **Python**:
    ```python
    from sklearn.metrics import log_loss
    log_loss(y_true, y_prob)
    ```

- **Equal Error Rate (EER)** (Used in biometric systems and verification tasks)

  - **Explanation**: The point where the false acceptance rate (FAR) equals the false rejection rate (FRR).
  - **Python**:
    ```python
    # Custom computation using ROC or DET curve intersection
    ```

---
### Regression Models

- **Mean Squared Error (MSE)** <span style="color:#DC143C">(Sensitive to outliers)</span>
  - **Explanation**: Average of squared errors.
  - **Formula**: $\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $
  - **Python**:
    ```python
    from sklearn.metrics import mean_squared_error
    mean_squared_error(y_true, y_pred)
    ```

- **Root Mean Squared Error (RMSE)** <span style="color:#6A5ACD">(Good for interpretability)</span>
  - **Explanation**: Square root of MSE.
  - **Python**:
    ```python
    import numpy as np
    np.sqrt(mean_squared_error(y_true, y_pred))
    ```

- **Mean Absolute Error (MAE)** <span style="color:#3CB371">(Robust to outliers)</span>
  - **Explanation**: Average of absolute differences.
  - **Formula**: $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i| $
  - **Python**:
    ```python
    from sklearn.metrics import mean_absolute_error
    mean_absolute_error(y_true, y_pred)
    ```

- **R² Score** <span style="color:#1E90FF">(Useful for overall fit)</span>
  - **Explanation**: Proportion of variance explained.
  - **Formula**: $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$
  - **Python**:
    ```python
    from sklearn.metrics import r2_score
    r2_score(y_true, y_pred)
    ```

- **Adjusted R²** <span style="color:#B8860B">(Better for feature-rich models)</span>
  - **Explanation**: Adjusts R² for number of predictors.
  - **Formula**: $1 - \left(1 - R^2\right) \cdot \frac{n - 1}{n - p - 1} $

- **Mean Absolute Percentage Error (MAPE)** <span style="color:#FF8C00">(Useful when scale matters)</span>
  - **Formula**: $\frac{100\%}{n} \sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| $
  - **Python**:
    ```python
    np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ```

---
### Clustering Models

- **Silhouette Score**:  
  Measures how similar each sample is to its own cluster (cohesion) compared to other clusters (separation).  
  <span style="color:#32CD32">(Good for evaluating cluster separation)</span>  
  **Formula**:  
  For a sample $ i $:
  $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
  where $ a(i) $ is the average distance to points in the same cluster, and $ b(i) $ is the minimum average distance to points in other clusters.

  ```python
  from sklearn.metrics import silhouette_score
  score = silhouette_score(X, labels)
  ```

- **Calinski-Harabasz Index**:  
  Evaluates the ratio of between-cluster dispersion to within-cluster dispersion.  
  <span style="color:#00BFFF">(Useful for overall cluster quality)</span>  
  **Formula**:
 $\text{CH} = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \cdot \frac{N - k}{k - 1}$
  where $ B_k $ and $ W_k $ are between and within-cluster dispersion matrices, $ N $ is the number of samples, and $ k $ is the number of clusters.

  ```python
  from sklearn.metrics import calinski_harabasz_score
  score = calinski_harabasz_score(X, labels)
  ```

- **Davies-Bouldin Index**:  
  Measures the average similarity between each cluster and the most similar one (lower is better).  
  <span style="color:#BA55D3">(Lower is better)</span>  
  **Formula**:
 $\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$
  where $ \sigma_i $ is the average distance of all points in cluster $ i $ to centroid $ c_i $, and $ d(c_i, c_j) $ is the distance between centroids.

  ```python
  from sklearn.metrics import davies_bouldin_score
  score = davies_bouldin_score(X, labels)
  ```

- **Adjusted Rand Index (ARI)**:  
  Measures similarity between predicted clustering and ground truth. Adjusted for chance.  
  <span style="color:#F08080">(Useful with labeled data)</span>  
  **Formula**:
 $\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$
  where RI is the Rand Index and $ \mathbb{E}[\text{RI}] $ is its expected value.

  ```python
  from sklearn.metrics import adjusted_rand_score
  score = adjusted_rand_score(true_labels, predicted_labels)
  ```


---
### Ranking Models

- **Normalized Discounted Cumulative Gain (NDCG)**:  
  Measures ranking quality by comparing the actual ranking to an ideal one. Rewards placing relevant items higher.  
  <span style="color:#FF69B4">(Good for relevance-sensitive tasks)</span>  
  **Formula**:  
  $\text{NDCG@k} = \frac{DCG@k}{IDCG@k}, \quad \text{where } DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$
  - $ rel_i $: relevance of the item at position $ i $  
  - $ IDCG@k $: DCG of the ideal ranking (used for normalization)

  **Python Example**:
  ```python
  from sklearn.metrics import ndcg_score

  y_true = [[3, 2, 3, 0, 1]]
  y_score = [[0.9, 0.8, 0.7, 0.1, 0.2]]
  score = ndcg_score(y_true, y_score, k=5)
  print(f"NDCG@5: {score:.3f}")
  ```

- **Mean Reciprocal Rank (MRR)**:  
  Evaluates the rank position of the first relevant item in a list. Higher is better.  
  <span style="color:#20B2AA">(Useful for top-k recommendations)</span>  
  **Formula**:  
  $\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$
  where $ \text{rank}_i $ is the position of the first relevant item for query $ i $, and $ N $ is the number of queries.

  **Python Example**:
  ```python
  def mean_reciprocal_rank(y_true, y_pred):
      ranks = []
      for true, pred in zip(y_true, y_pred):
          for rank, item in enumerate(pred, start=1):
              if item in true:
                  ranks.append(1 / rank)
                  break
          else:
              ranks.append(0)
      return sum(ranks) / len(ranks)

  y_true = [[2], [1], [3]]
  y_pred = [[1, 2, 3], [2, 1, 3], [3, 1, 2]]
  score = mean_reciprocal_rank(y_true, y_pred)
  print(f"MRR: {score:.3f}")
  ```

- **Mean Average Precision (MAP)** (Useful for ranking in retrieval and recommendation)
  - **Explanation**: Average of precision values at ranks where relevant items appear, averaged over all queries.
  - **Formula**: $\text{MAP} = \frac{1}{|Q|} \sum\_{q \in Q} \text{AP}(q)$
  - **Python**:
    ```python
    from sklearn.metrics import average_precision_score
    # Use per-query then average if multiple queries exist
    ```

## 2. Model Robustness
These techniques test how well a model holds up under different data or distributional conditions.

- **Cross-Validation**  
  Estimates model performance by partitioning data into training and testing sets multiple times.  
  <span style="color:#4682B4">(Helps assess generalization performance)</span>  
  **Concept**: Divide data into *k* folds, train on *k−1* folds and test on the remaining fold, then average the results.

  **Python Example**:
  ```python
  from sklearn.model_selection import cross_val_score
  from sklearn.ensemble import RandomForestClassifier

  scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
  print(f"Cross-Validation Accuracy: {scores.mean():.3f}")
  ```


- **Bootstrapping**  
  Resamples the dataset with replacement to estimate performance variance or confidence intervals.  
  <span style="color:#DAA520">(Evaluates model stability and variability)</span>  
  **Concept**: Train the model on multiple bootstrapped datasets and analyze result variability.

  **Python Example**:
  ```python
  import numpy as np
  from sklearn.utils import resample
  from sklearn.metrics import accuracy_score

  boot_scores = []
  for _ in range(100):
      X_resampled, y_resampled = resample(X, y)
      model.fit(X_resampled, y_resampled)
      y_pred = model.predict(X_test)
      boot_scores.append(accuracy_score(y_test, y_pred))

  print(f"Bootstrap Mean Accuracy: {np.mean(boot_scores):.3f}")
  ```


- **Adversarial Testing**  
  Tests model robustness by generating small perturbations that fool the model.  
  <span style="color:#FF6347">(Tests sensitivity to subtle input changes)</span>  
  **Concept**: Use gradient-based or heuristic methods to slightly alter inputs and observe changes in output.

  **Python Example**:
  ```python
  import torch
  import torch.nn.functional as F

  def fgsm_attack(data, epsilon, gradient):
      return data + epsilon * gradient.sign()

  # Requires a PyTorch model and data
  # This is just a conceptual placeholder
  ```


- **Out-of-Distribution (OOD) Testing**  
  Evaluates model behavior on data that comes from a different distribution than the training set.  
  <span style="color:#8A2BE2">(Checks generalization beyond training domain)</span>  
  **Concept**: Compare model confidence and accuracy between in-distribution and OOD data.

  **Python Example**:
  ```python
  # Inference on OOD samples
  in_preds = model.predict(in_distribution_X)
  ood_preds = model.predict(ood_X)

  print("In-Distribution Accuracy:", accuracy_score(in_distribution_y, in_preds))
  print("OOD Prediction Confidence (mean):", np.mean(np.max(ood_preds, axis=1)))
  ```

## 3. Model Interpretability  
These methods help us **understand why a model makes certain predictions**, increasing transparency and trust.


- **Feature Importance (Permutation, SHAP, LIME)**  
  Measures how much each feature contributes to the prediction.

  **Permutation Importance Formula**:  
  $\text{Importance}(f) = \text{score}_{\text{base}} - \text{score}_{f \text{ shuffled}}$  
  **Python Example (Permutation)**:
  ```python
  from sklearn.inspection import permutation_importance
  result = permutation_importance(model, X_val, y_val, n_repeats=10)
  print(result.importances_mean)
  ```

- **Partial Dependence Plots (PDPs)**  
  Shows the average model prediction as a feature varies, keeping all others constant.  
  <span style="color:#4169E1">(Visualizes global feature effect)</span>  

  **Python Example**:
  ```python
  from sklearn.inspection import PartialDependenceDisplay
  PartialDependenceDisplay.from_estimator(model, X, features=[0, 1])
  ```

- **SHAP (SHapley Additive exPlanations)**  
  Based on cooperative game theory, assigns each feature a "fair" contribution to a prediction.

  **Formula (Shapley Value)**:
  $\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$

  **Python Example**:
  ```python
  import shap
  explainer = shap.Explainer(model.predict, X)
  shap_values = explainer(X)
  shap.plots.beeswarm(shap_values)
  ```


- **LIME (Local Interpretable Model-agnostic Explanations)**  
  Explains a single prediction by approximating the model locally with an interpretable one.

  **Concept**: Fit a simple model to perturbed inputs around the target sample.

  **Python Example**:
  ```python
  import lime
  from lime.lime_tabular import LimeTabularExplainer

  explainer = LimeTabularExplainer(X_train.values, mode='classification', feature_names=X.columns)
  exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
  exp.show_in_notebook()
  ```


- **Attention Maps (for Neural Networks)**  
  Visualizes which input parts the model focuses on—often used in NLP and vision tasks.

  **Concept**: Extract attention weights from model layers.

  **Python Example (Transformers)**:
  ```python
  from transformers import BertTokenizer, BertModel
  import torch

  model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
  inputs = tokenizer("Example sentence", return_tensors="pt")
  outputs = model(**inputs)
  attention = outputs.attentions[-1]  # Last layer attention
  ```

