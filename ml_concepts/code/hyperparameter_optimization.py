import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Define objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
    )

    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return 1.0 - score  # Minimize 1 - accuracy

# Create and run study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best params
print("Best hyperparameters:", study.best_params)

# Train final model
final_model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(**study.best_params, random_state=42)
)
final_model.fit(X, y)

# 📈 Visualization
ax1 = optuna.visualization.matplotlib.plot_optimization_history(study)
ax1.set_title("Optimization History")

ax2 = optuna.visualization.matplotlib.plot_param_importances(study)
ax2.set_title("Hyperparameter Importance")

# Pick two important parameters to visualize in contour
param_x = 'n_estimators'
param_y = 'max_depth'
param_z = 'min_samples_leaf'

ax3 = optuna.visualization.matplotlib.plot_contour(study, params=[param_x, param_y])
ax3.set_title(f"Contour of {param_x} vs {param_y}")

ax4 = optuna.visualization.matplotlib.plot_contour(study, params=[param_x, param_z])
ax4.set_title(f"Contour of {param_x} vs {param_z}")

# Best values
best_x = study.best_trial.params[param_x]
best_y = study.best_trial.params[param_y]
best_z = study.best_trial.params[param_z]
best_val = study.best_trial.value

ax3.plot(best_x, best_y, 'ro', label='Best Trial')
ax3.legend()

ax4.plot(best_x, best_z, 'ro', label='Best Trial')
ax4.legend()

plt.show()
