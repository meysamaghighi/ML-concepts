## ML Questions
1. How will you detect overfitting in neural networks?
    1. During training? How will you detect if you are overfitting?
2. If you keep training a NN, what will happen to training and evaluation loss? (beyond the saturation point)
3. How will you build a model to classify sentences?
    1. What loss function will you use?
    2. What if you want word permutations to not impact the outcome?
4. Do you have experience with generative AI?



## Responses (ChatGPT):

### **1. How will you detect overfitting in neural networks?**
#### **1.1 During training? How will you detect if you are overfitting?**
- **Answer:**
  Overfitting happens when your model performs well on training data but poorly on validation/test data.
  - **During training**, track both **training loss** and **validation loss**:
    - If **training loss decreases** while **validation loss increases**, you are overfitting.
    - Also, monitor **metrics** like accuracy/F1 on train vs val/test sets—if there's a large gap, it's another indicator.
  - **Visualization** (loss curves) helps a lot: look for divergence between the two losses.

---

### **2. If you keep training a neural network, what will happen to training and evaluation loss (beyond saturation point)?**
- **Answer:**
  - **Training loss** will generally **keep decreasing** (or plateau at a very low value).
  - **Evaluation (validation/test) loss**:
    - Initially, it may decrease (model is generalizing better).
    - Eventually, it **starts increasing** due to **overfitting**—the model starts to memorize the training data and loses generalizability.
  - This is why we often use **early stopping** based on validation loss.

---

### **3. How will you build a model to classify sentences?**
- **Answer:**
  - General pipeline:
    1. **Text preprocessing** (tokenization, padding/truncation, etc.)
    2. Use an embedding method:
       - Classical: TF-IDF or word2vec.
       - Deep: Pretrained models like BERT or sentence transformers.
    3. Feed into a classifier:
       - Simple: MLP.
       - Advanced: Transformer-based models with fine-tuning.
    4. Use labeled dataset with sentences and class labels.

#### **3.1 What loss function will you use?**
- **Answer:**
  - Depends on the task:
    - **Binary classification**: `Binary Cross-Entropy`
    - **Multiclass classification**: `Categorical Cross-Entropy`
    - **Multi-label**: `Binary Cross-Entropy` with sigmoid outputs
  - For deep models, usually softmax + cross-entropy is standard.

#### **3.2 What if you want word permutations to not impact the outcome?**
- **Answer:**
  You want the model to be **permutation-invariant** to word order. Options:
  - Use **Bag-of-Words** or **TF-IDF**: Ignores word order.
  - Use **average pooling over embeddings**: Treats words equally.
  - For deep models:
    - Use **self-attention mechanisms** (like in Transformers) and **train the model** to focus on semantic meaning rather than word order.
    - Or use architectures like **DeepSets** for true permutation invariance (more common in set-based inputs).

---

### **4. Do you have experience with generative AI?**
- **Sample tailored answer (if you do):**
  > Yes, I've worked with generative models like **VAEs**, **GANs**, and more recently **transformer-based LLMs** for both generation and fine-tuning. For example, I’ve explored [text/image/code] generation using models like **GPT**, **T5**, or **Stable Diffusion**, and also built prompt pipelines for task-specific generation.
  
- **If you haven't directly worked with it:**
  > I’ve followed recent work in generative AI closely and understand the core models (GANs, VAEs, LLMs like GPT), and while I haven’t built a production-scale generative model, I’ve experimented with them in side projects and would be excited to apply that in practice.

---

Let me know if you want me to help write ideal answers in STAR format (situation, task, action, result) for behavioral interviews or want clarifications or references for any of the answers.