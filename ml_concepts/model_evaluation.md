# Machine Learning Model Evaluation Methods

Analyzing and evaluating a machine learning (ML) model involves assessing its performance, robustness, fairness, and interpretability. Below is a comprehensive list of evaluation methods, categorized based on what they measure.

## 1. Performance Metrics
These measure how well a model makes predictions.

### Classification Models
- **Accuracy**: The proportion of correct predictions. <span style="color:#90EE90">(Best for balanced datasets)</span>
- **Precision**: The proportion of true positive predictions among all positive predictions. <span style="color:#FFD700">(Useful when false positives are costly)</span>
- **Recall (Sensitivity)**: The proportion of true positives among all actual positives. <span style="color:#00CED1">(Useful when false negatives are costly)</span>
- **F1 Score**: The harmonic mean of precision and recall. <span style="color:#FF6347">(Useful when both false positives and false negatives matter)</span>
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the trade-off between true positive rate and false positive rate. <span style="color:#DA70D6">(Useful for imbalanced datasets)</span>
- **PR-AUC (Precision-Recall Area Under Curve)**: Evaluates models on imbalanced datasets when positive class is rare. <span style="color:#FFA07A">(Useful when positive class is rare)</span>
- **Log Loss (Cross-Entropy Loss)**: Measures how well probability estimates align with actual labels. <span style="color:#7FFFD4">(Useful for probabilistic models)</span>

### Regression Models
- **Mean Squared Error (MSE)**: Penalizes larger errors more than smaller ones. <span style="color:#DC143C">(Sensitive to outliers)</span>
- **Root Mean Squared Error (RMSE)**: The square root of MSE, interpretable in original units. <span style="color:#6A5ACD">(Good for interpretability)</span>
- **Mean Absolute Error (MAE)**: Measures absolute differences, less sensitive to outliers than MSE. <span style="color:#3CB371">(Robust to outliers)</span>
- **R² Score (Coefficient of Determination)**: Measures how well predictions explain variance in data. <span style="color:#1E90FF">(Useful for overall fit)</span>
- **Adjusted R²**: Adjusts for number of features in a regression model. <span style="color:#B8860B">(Better for feature-rich models)</span>
- **Mean Absolute Percentage Error (MAPE)**: Measures percentage deviation between predicted and actual values. <span style="color:#FF8C00">(Useful when scale matters)</span>

### Clustering Models
- **Silhouette Score**: Measures how similar a sample is to its own cluster compared to other clusters. <span style="color:#32CD32">(Good for evaluating cluster separation)</span>
- **Calinski-Harabasz Index**: Evaluates cluster density and separation. <span style="color:#00BFFF">(Useful for overall cluster quality)</span>
- **Davies-Bouldin Index**: Measures cluster compactness and separation. <span style="color:#BA55D3">(Lower is better)</span>
- **Adjusted Rand Index (ARI)**: Measures the similarity between predicted and true clustering. <span style="color:#F08080">(Useful with labeled data)</span>

### Ranking Models
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality, higher scores mean better ranking. <span style="color:#FF69B4">(Good for relevance-sensitive tasks)</span>
- **Mean Reciprocal Rank (MRR)**: Evaluates how quickly relevant items appear in a ranked list. <span style="color:#20B2AA">(Useful for top-k recommendations)</span>

## 2. Model Robustness
These techniques test a model’s reliability under different conditions.

- **Cross-Validation** (K-Fold, Stratified K-Fold, Leave-One-Out, etc.): Helps assess generalization performance.
- **Bootstrapping**: Evaluates variability by resampling data with replacement.
- **Adversarial Testing**: Exposes models to adversarially perturbed inputs.
- **Out-of-Distribution (OOD) Testing**: Evaluates model behavior on unseen data distributions.

## 3. Model Interpretability
These methods help understand model decisions.

- **Feature Importance (Permutation, SHAP, LIME)**: Identifies key features affecting predictions.
- **Partial Dependence Plots (PDPs)**: Shows how a feature affects predictions.
- **SHAP (SHapley Additive exPlanations)**: Provides feature attribution for individual predictions.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains specific predictions by perturbing inputs.
- **Attention Maps (for Neural Networks)**: Visualizes which input features are important for a prediction.

## 4. Fairness & Bias Analysis
Ensures a model does not introduce unwanted bias.

- **Demographic Parity**: Checks if predictions are independent of sensitive attributes.
- **Equalized Odds**: Ensures equal true positive and false positive rates across groups.
- **Counterfactual Fairness**: Tests if changing a sensitive attribute changes the prediction.

## 5. Computational Efficiency
Assesses model speed and resource usage.

- **Inference Latency**: Measures prediction time.
- **Memory Footprint**: Evaluates how much RAM the model requires.
- **Compute Cost**: Analyzes FLOPS, energy usage, and cost of inference.

## 6. Business & Domain-Specific Metrics
Tailored metrics for specific applications.

- **Churn Prediction** (<span style="color:#7CFC00">Lift & Gains Chart, Profit Curve</span>)
- **Fraud Detection** (<span style="color:#FF4500">Precision-Recall Curve, F1 Score</span>)
- **Recommendation Systems** (<span style="color:#00FA9A">Hit Rate, Coverage, Novelty</span>)

