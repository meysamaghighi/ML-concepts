## 🚀 Top Neural Network Optimizers

| Optimizer      | Type            | Key Features                                                                 |
|----------------|-----------------|------------------------------------------------------------------------------|
| **SGD**        | First-order     | Basic optimizer; can be enhanced with momentum or learning rate schedules.  |
| **Momentum**   | First-order     | Adds momentum term to smooth updates and escape local minima.               |
| **Nesterov**   | First-order     | Looks ahead at future position before computing gradient.                   |
| **Adam**       | Adaptive        | Combines Momentum + RMSProp; default for many deep learning models.         |
| **RMSProp**    | Adaptive        | Scales learning rate by moving average of recent squared gradients.         |
| **Adagrad**    | Adaptive        | Adapts learning rate based on how frequently parameters get updated.        |
| **Adadelta**   | Adaptive        | Like Adagrad but solves the learning rate decay problem.                    |
| **AdamW**      | Adaptive        | Adam with decoupled weight decay (used in BERT, transformers).              |
| **LAMB**       | Layer-wise      | Scales updates by layer norm; used for very large models like GPT.          |
| **Lion**       | Momentum-based  | Lightweight optimizer proposed by Google (used in vision transformers).     |

---

## 🧰 Optimizers in Major Libraries

### 🔵 **PyTorch (`torch.optim`)**
```python
import torch.optim as optim

optim.SGD(...)
optim.Adam(...)
optim.AdamW(...)
optim.RMSprop(...)
optim.Adagrad(...)
```

Also includes:
- `LBFGS` (for small problems or fine-tuning)
- `ASGD` (averaged SGD for convex problems)
- Third-party: `LAMB`, `Lion` via `bitsandbytes`, `timm`, etc.

---

### 🟢 **TensorFlow / Keras (`tf.keras.optimizers`)**
```python
import tensorflow as tf

tf.keras.optimizers.SGD(...)
tf.keras.optimizers.Adam(...)
tf.keras.optimizers.RMSprop(...)
tf.keras.optimizers.Adagrad(...)
tf.keras.optimizers.AdamW(...)
```

Also includes:
- `Nadam` (Adam + Nesterov)
- `Ftrl` (used in linear models with large feature spaces)
- Optimizer wrappers for learning rate schedules, weight decay

---

### 🔬 When to Use What?

| Scenario                                  | Recommended Optimizer       |
|------------------------------------------|------------------------------|
| Most deep learning models (default)      | ✅ **Adam**                  |
| Image models (CNNs)                      | ✅ SGD + Momentum, Adam      |
| Transformers / NLP                       | ✅ AdamW, LAMB               |
| Low memory / edge devices                | ✅ RMSProp, Adagrad          |
| Very large models (e.g., GPT)            | ✅ LAMB, Lion                |
| Simple linear models                     | ✅ SGD                       |

