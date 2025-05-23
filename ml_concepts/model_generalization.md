# Generalization Techniques in Machine Learning 🎯📊✨

## Overview 
Generalization in Machine Learning ensures that models perform well on unseen data. Below are key techniques to improve generalization, their applicability, and common use cases.

## Generalization Methods 

Here’s the reordered table based on your request:

1. **Non-NN methods come first.**
2. Then **NN & Non-NN** methods.
3. Lastly, **NN-only** methods.
4. Within each group, more **common/widely-used** methods are ranked higher.

---

| **#** | **Method**                | **Description** | **Applicable Models (NN or not)** | **Use Cases** | **PyTorch Syntax** | **TensorFlow Syntax** |
|------:|---------------------------|-----------------|-----------------------------------|----------------|--------------------|------------------------|
| 1 | **Feature Engineering**   | Creates informative features to improve model performance. | Non-NN | Classical ML models | `sklearn.feature_selection.SelectKBest()` | `tf.feature_column.numeric_column()` |
| 2 | **Cross-Validation**      | Splits data into multiple subsets for training/validation to improve performance. | NN & Non-NN | Small datasets, Regression, Classification | `sklearn.model_selection.KFold(n_splits=5)` | `tf.data.experimental.sample_from_datasets()` |
| 3 | **Regularization (L1/L2)**| Adds penalty terms to loss function to prevent overfitting. | NN & Non-NN | Regression, Classification | `torch.nn.L1Loss()` / `torch.nn.MSELoss(weight_decay=0.01)` | `tf.keras.regularizers.l1(0.01)`, `tf.keras.regularizers.l2(0.01)` |
| 4 | **Early Stopping**        | Stops training when validation loss starts increasing. | NN & Non-NN | Any ML model | `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)` | `tf.keras.callbacks.EarlyStopping(patience=5)` |
| 5 | **Ensemble Learning**     | Combines multiple models to reduce variance and bias. | NN & Non-NN | Random Forest, Boosting, Stacking | `sklearn.ensemble.RandomForestClassifier()` | `tf.keras.models.clone_model()` |
| 6 | **Hyperparameter Tuning** | Optimizes model parameters for better generalization. | NN & Non-NN | All ML models | `Optuna: study.optimize(objective, n_trials=100)` | `KerasTuner: RandomSearch()` |
| 7 | **Data Augmentation**     | Generates variations of training data to improve model robustness. | NN & Non-NN | Image Classification, NLP | `torchvision.transforms.RandomHorizontalFlip()` | `tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)` |
| 8 | **Dropout**               | Randomly deactivates neurons during training to force robustness. | NN | Deep Learning (Image, NLP) | `torch.nn.Dropout(p=0.5)` | `tf.keras.layers.Dropout(0.5)` |
| 9 | **Batch Normalization**   | Normalizes activations to stabilize and accelerate training. | NN | Deep Learning (CNN, RNN) | `torch.nn.BatchNorm2d(num_features)` | `tf.keras.layers.BatchNormalization()` |
| 10 | **Transfer Learning**     | Uses pre-trained models to improve performance on new tasks. | NN | Image Recognition, NLP | `torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)` | `tf.keras.applications.ResNet50(weights='imagenet')` |
| 11 | **Weight Initialization** | Improves convergence and avoids vanishing/exploding gradients. | NN | Deep Learning | `torch.nn.init.xavier_uniform_(tensor)` | `tf.keras.initializers.GlorotUniform()` |
| 12 | **Gradient Clipping**     | Limits gradients to avoid instability. | NN | Deep Learning (RNN, Transformers) | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` | `tf.keras.optimizers.Adam(clipnorm=1.0)` |

---
