# Autoencoders
## 🌌 1. **Autoencoder**

An **autoencoder** is a type of neural network used to **learn efficient representations of data**, typically for **dimensionality reduction**, **denoising**, or **unsupervised feature learning**.

---

## 🧠 Core Idea

An autoencoder tries to **reconstruct its input** by first compressing it and then decompressing it:

```text
Input → [Encoder] → Latent Code → [Decoder] → Output (≈ Input)
```

---

## 🔧 Structure

### 1. **Encoder**:
- Compresses the input into a **lower-dimensional latent space** (also called the bottleneck or embedding).
- Learns the most important features in the input.

### 2. **Latent Code**:
- The compressed representation.
- Ideally, contains only the essential information to reconstruct the original input.

### 3. **Decoder**:
- Reconstructs the input from the latent code.
- Learns to reverse the encoding process.

---

## 🧪 PyTorch Example: Simple Autoencoder for Images

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: input 784 → 128 → 64
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Decoder: 64 → 128 → 784
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # values between 0 and 1
        )

    def forward(self, x):
        x = x.view(-1, 28*28)   # Flatten
        code = self.encoder(x)
        out = self.decoder(code)
        return out.view(-1, 1, 28, 28)  # Reshape back to image
```

---

## 🔍 Loss Function

```python
criterion = nn.MSELoss()
output = model(input)
loss = criterion(output, input)  # we want output ≈ input
```

---

## 🧰 Use Cases

| Task                      | How Autoencoders Help                |
|---------------------------|--------------------------------------|
| **Dimensionality Reduction** | Learn compressed features (like PCA but nonlinear) |
| **Denoising**             | Remove noise from input by training on clean data |
| **Anomaly Detection**     | Reconstruction error is high for unseen or anomalous data |
| **Pretraining**           | Learn useful features for other models |

---

Awesome, Meysam! Let's explore both **Variational Autoencoders (VAEs)** and **Convolutional Autoencoders (CAEs)** — two very powerful extensions of the basic autoencoder.

---

## 🌌 2. **Variational Autoencoder (VAE)**

A **Variational Autoencoder** is a **probabilistic version** of an autoencoder. Instead of learning a fixed latent vector, the VAE learns a **distribution over the latent space**.

### 🧠 Key Concepts

- The encoder doesn't output just a single latent vector, but **two vectors**:
  - **Mean (μ)** and **log-variance (logσ²)** of a Gaussian distribution.
- During training, we **sample** from this distribution using the **reparameterization trick**.

### 🔁 Why?

This gives us:
- A **continuous, smooth latent space**
- The ability to **generate new samples** by sampling from the latent distribution

---

### 🔢 VAE Loss

VAE loss =  
**Reconstruction Loss** (like MSE)  
+  
**KL Divergence** (regularizes the latent space to follow a standard normal distribution)

---

### 🔧 PyTorch VAE Example (Simplified)

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, 20)  # μ
        self.fc22 = nn.Linear(256, 20)  # logσ²
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z = μ + σ * ε

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

---

### 📉 VAE Loss Function

```python
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
```

---

## 🖼️ 3. **Convolutional Autoencoder (CAE)**

A **Convolutional Autoencoder** is designed specifically for **image data**. It replaces the linear layers with **convolutional** and **deconvolutional** (transposed convolution) layers.

---

### 🧱 Architecture

- **Encoder**:
  - Convolution + ReLU + Pooling → feature maps
- **Decoder**:
  - Transposed convolution (or upsampling) → reconstruct original image

---

### 🔧 PyTorch CAE Example

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14x14 → 7x7
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 14x14 → 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

---

### ✅ When to Use Which?

| Type                     | Use Case                                 |
|--------------------------|-------------------------------------------|
| **Standard Autoencoder** | Simple denoising, dimensionality reduction |
| **VAE**                  | Data generation, structured latent space  |
| **CAE**                  | Image-based compression or denoising      |

---




# Training Examples

Training examples for both a **VAE** and a **Convolutional Autoencoder (CAE)** using the **MNIST** dataset in PyTorch. These will include data loading, model definition, training loop, and visualization steps.

---

## 🖼️ Part 1: Training a **VAE on MNIST**

### 🔌 Step 1: Import Libraries and Load Data

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

---

### 🧠 Step 2: Define the VAE Model

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, 20)  # μ
        self.fc22 = nn.Linear(256, 20)  # logσ²
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

---

### 📉 Step 3: Define the Loss Function

```python
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

---

### 🚀 Step 4: Train the VAE

```python
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for epoch in range(10):
    total_loss = 0
    for x, _ in train_loader:
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f\"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}\")
```

---

### 👁️ Step 5: Visualize Reconstructed Images

```python
model.eval()
with torch.no_grad():
    x, _ = next(iter(train_loader))
    recon, _, _ = model(x)
    recon = recon.view(-1, 1, 28, 28)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x[0][0], cmap='gray')
    axs[0].set_title(\"Original\")
    axs[1].imshow(recon[0][0], cmap='gray')
    axs[1].set_title(\"Reconstructed\")
    plt.show()
```

---

## 🧱 Part 2: Training a **Convolutional Autoencoder (CAE)** on MNIST

### 🧠 Step 1: Define CAE Model

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

---

### 🚀 Step 2: Train CAE

```python
model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    total_loss = 0
    for x, _ in train_loader:
        recon = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f\"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}\")
```

---

### 👁️ Step 3: Visualize CAE Output

```python
model.eval()
with torch.no_grad():
    x, _ = next(iter(train_loader))
    recon = model(x)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x[0][0], cmap='gray')
    axs[0].set_title(\"Original\")
    axs[1].imshow(recon[0][0], cmap='gray')
    axs[1].set_title(\"Reconstructed\")
    plt.show()
```

---
