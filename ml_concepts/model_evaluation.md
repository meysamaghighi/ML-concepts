# Machine Learning Model Evaluation Methods

Analyzing and evaluating a machine learning (ML) model involves assessing its performance, robustness, fairness, and interpretability. Below is a comprehensive list of evaluation methods, categorized based on what they measure.

## 1. Performance Metrics
These measure how well a model makes predictions.

### Classification Models
- **Accuracy**: The proportion of correct predictions. *(Best for balanced datasets)*
- **Precision**: The proportion of true positive predictions among all positive predictions. *(Useful when false positives are costly)*
- **Recall (Sensitivity)**: The proportion of true positives among all actual positives. *(Useful when false negatives are costly)*
- **F1 Score**: The harmonic mean of precision and recall. *(Useful when both false positives and false negatives matter)*
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the trade-off between true positive rate and false positive rate. *(Useful for imbalanced datasets)*
- **PR-AUC (Precision-Recall Area Under Curve)**: Evaluates models on imbalanced datasets when positive class is rare.
- **Log Loss (Cross-Entropy Loss)**: Measures how well probability estimates align with actual labels.

### Regression Models
- **Mean Squared Error (MSE)**: Penalizes larger errors more than smaller ones.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, interpretable in original units.
- **Mean Absolute Error (MAE)**: Measures absolute differences, less sensitive to outliers than MSE.
- **R² Score (Coefficient of Determination)**: Measures how well predictions explain variance in data.
- **Adjusted R²**: Adjusts for number of features in a regression model.
- **Mean Absolute Percentage Error (MAPE)**: Measures percentage deviation between predicted and actual values.

### Clustering Models
- **Silhouette Score**: Measures how similar a sample is to its own cluster compared to other clusters.
- **Calinski-Harabasz Index**: Evaluates cluster density and separation.
- **Davies-Bouldin Index**: Measures cluster compactness and separation.
- **Adjusted Rand Index (ARI)**: Measures the similarity between predicted and true clustering.

### Ranking Models
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality, higher scores mean better ranking.
- **Mean Reciprocal Rank (MRR)**: Evaluates how quickly relevant items appear in a ranked list.

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

- **Churn Prediction** (Lift & Gains Chart, Profit Curve)
- **Fraud Detection** (Precision-Recall Curve, F1 Score)
- **Recommendation Systems** (Hit Rate, Coverage, Novelty)
