## 🔧 Hyperparameters to Optimize in ML

---

### ✅ 1. Traditional ML Models

These include algorithms like **Logistic Regression**, **SVM**, **Random Forests**, **XGBoost**, etc.

| Model               | Key Hyperparameters |
|--------------------|---------------------|
| **Logistic Regression** | Regularization strength (`C`), penalty (`l1`/`l2`) |
| **SVM**                  | Kernel type, `C` (margin), `gamma` (RBF kernel) |
| **k-NN**                 | Number of neighbors (`k`), distance metric |
| **Decision Trees**       | Max depth, min samples per leaf, criterion |
| **Random Forest**        | Number of trees, max depth, max features |
| **XGBoost / LightGBM**   | Learning rate (`eta`), `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, `lambda`, `alpha` |

---

### ✅ 2. Modern Deep Learning Models (e.g., CNNs, RNNs, Transformers)

| Category | Common Hyperparameters |
|----------|------------------------|
| **Architecture** | Number of layers, layer types, hidden units, activation functions |
| **Optimization** | Learning rate, optimizer type (Adam, SGD), momentum, weight decay |
| **Regularization** | Dropout rate, L2 penalty, batch norm usage |
| **Training** | Batch size, number of epochs, early stopping patience |
| **Initialization** | Weight initialization method |
| **Learning Rate Scheduling** | Warmup steps, decay schedule (e.g., cosine, step) |

📌 *Learning rate* is usually the most sensitive hyperparameter!

---

### ✅ 3. Generative AI / Large Language Models (LLMs, Diffusion, etc.)

#### 🔹 Pretraining Phase:
| Hyperparameter            | Role |
|---------------------------|------|
| Model size (depth, width) | Controls capacity and cost |
| Sequence length           | Affects attention context |
| Vocabulary size           | Tokenization granularity |
| Batch size                | Stability vs memory |
| Learning rate + warmup    | Critical for stable training |
| Optimizer (AdamW, etc.)   | Handles sparse updates well |
| Weight decay              | Regularizes large models |

#### 🔹 Fine-tuning / Prompt-Tuning:
| Hyperparameter            | Role |
|---------------------------|------|
| Learning rate             | Crucial for LoRA, adapters |
| Number of steps           | Controls overfitting |
| Prompt length             | Longer prompts = more context |
| Temperature               | Controls randomness in generation |
| Top-k / Top-p sampling    | Controls diversity of outputs |
| Max tokens / stop tokens  | Output size control |

#### 🔹 Diffusion Models (e.g., for image generation):
| Hyperparameter            | Role |
|---------------------------|------|
| Number of diffusion steps | Affects quality and compute |
| Noise schedule            | Controls denoising behavior |
| Beta schedule (linear, cosine) | For diffusion process |
| Guidance scale            | Tradeoff between realism and diversity |
| Sampling algorithm        | DDIM, ancestral sampling, etc. |

---

### ✅ Cross-Cutting Techniques

| Hyperparameter Type | Examples |
|---------------------|----------|
| **Regularization**  | Dropout, L2 penalty, early stopping |
| **Learning Rate**   | Static, OneCycle, Cosine decay |
| **Data Augmentation** | Flip, crop, cutout, mixup |
| **Loss Functions**  | CE, Focal, Contrastive, Triplet, MSE |
| **Gradient Clipping** | Prevent exploding gradients |
| **Mixed Precision** | Precision level (fp32 vs fp16) |

---

### 🧠 Notes on Optimization Strategies

- **Grid Search**: Exhaustive, slow
- **Random Search**: Often better for high-dimensional spaces
- **Bayesian Optimization**: Uses prior to explore smartly
- **Hyperband / Optuna**: Early stopping + sampling
- **Population-Based Training (PBT)**: Used in RL and GenAI

---
