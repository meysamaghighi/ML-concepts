### 🌟 Classifier Comparison Table (with Time & Space Complexity)

Classifiers **sorted by their training time complexity from lowest to highest** (minimum O(n x d) is needed for storing input data):

| Time Complexity (Train)  | Space Complexity        | Classifier                             | Type                  | `sklearn` / Lib Name                   |
|--------------------------|--------------------------|----------------------------------------|-----------------------|----------------------------------------|
| O(n × d)                | O(d)                     | Naive Bayes                            | Probabilistic         | `MultinomialNB()`                      |
| O(n × d)                | O(d)                     | Logistic Regression                    | Linear Model          | `LogisticRegression()`                 |
| O(k × n × d)            | O(d)                     | SGDClassifier                          | Linear Model          | `SGDClassifier()`                      |
| O(n × d)                | O(n × d)                 | KNN Classifier                         | Instance-Based        | `KNeighborsClassifier()`               |
| O(n × d × log n)        | O(n)                     | Decision Tree                          | Tree-Based            | `DecisionTreeClassifier()`             |
| O(n × d²)               | O(d²)                    | Quadratic Discriminant Analysis (QDA)  | Probabilistic         | `QuadraticDiscriminantAnalysis()`      |
| O(t × n × d)            | O(t × d)                 | XGBoost                                | Ensemble (Boost)      | `XGBClassifier()`                      |
| O(t × n × d)            | O(t × d)                 | CatBoost                               | Ensemble (Boost)      | `CatBoostClassifier()`                 |
| O(t × n × log d)        | O(t × d)                 | LightGBM                               | Ensemble (Boost)      | `LGBMClassifier()`                     |
| O(t × n × log n)        | O(t × n)                 | Extra Trees                            | Ensemble (Tree-Based) | `ExtraTreesClassifier()`               |
| O(t × n × d × log n)    | O(t × n)                 | Random Forest                          | Ensemble (Tree-Based) | `RandomForestClassifier()`             |
| O(e × n × d × h)        | O(h × d)                 | MLPClassifier (Neural Net)             | Neural Network        | `MLPClassifier()`                      |
| O(n² × d)               | O(n²)                    | SVC (RBF Kernel)                       | Kernel Method         | `SVC(kernel='rbf')`                    |
| O(n³)                   | O(n²)                    | Gaussian Process Classifier            | Probabilistic         | `GaussianProcessClassifier()`          |
| Depends on base models  | Sum of base models       | Voting Classifier                      | Ensemble (Voting)     | `VotingClassifier()`                   |
| Depends on all models   | Sum of all models        | Stacking Classifier                    | Ensemble (Stacking)   | `StackingClassifier()`                 |

---

### 📘 Parameter Definitions

| Symbol | Meaning |
|--------|---------|
| **n** | Number of training samples |
| **d** | Number of features (dimensionality of input) |
| **k** | Number of iterations (epochs) for SGD |
| **t** | Number of trees or estimators |
| **e** | Number of training epochs (for neural nets) |
| **h** | Number of hidden units in the neural network |

---

### 🕐 Notes on Complexity

- The listed **time complexity** is for **training** unless otherwise noted.
- For most models, **inference time** is **much faster** than training:
  - For example, **KNN** is fast to train (no training!) but **slow to predict**: \( O(n × d) \)
  - In contrast, **tree-based models** like Random Forest or XGBoost have **fast inference** (typically \( O(\text{depth}) \))
- Models like **SGDClassifier** can support **online learning**, making them very efficient on streaming data.

