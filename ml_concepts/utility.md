## 🧰 Utility Tools & Model Workflow Helpers

| Tool/Concept         | Description                                  | Use Case                                           | Example Code                                  | Notes                             |
|----------------------|----------------------------------------------|---------------------------------------------------|-----------------------------------------------|-----------------------------------|
| `Pipeline`           | Combine preprocessing + modeling             | Clean, reusable workflows                         | `Pipeline([('scaler', ...), ('clf', ...)])`   | Use with `GridSearchCV`          |
| `GridSearchCV`       | Exhaustive search over hyperparams           | Best accuracy with known grid                     | `GridSearchCV(model, param_grid)`             | Slower than random search         |
| `RandomizedSearchCV` | Random sampling of param grid                | Faster tuning for large spaces                    | `RandomizedSearchCV(model, param_distributions)`| Tradeoff speed vs. thoroughness   |
| `ColumnTransformer`  | Apply different transforms by column         | Mixed feature types (e.g., num + cat)             | `ColumnTransformer([...])`                    | Essential for structured data     |
| `FeatureUnion`       | Combine multiple feature extractors          | NLP + numeric pipelines                           | `FeatureUnion([('text', ...), ('meta', ...)])` | Parallel preprocessing            |
| `StandardScaler`     | Scale to mean=0, std=1                       | Needed for linear models, SVMs                    | `StandardScaler()`                            | Use in pipelines                  |
| `OneHotEncoder`      | Encode categorical features                  | Convert strings to numeric vectors                | `OneHotEncoder()`                             | Combine with `ColumnTransformer`  |
| `PolynomialFeatures` | Generate polynomial & interaction features   | Capture non-linearity                             | `PolynomialFeatures(degree=2)`                | Combine with linear models        |
