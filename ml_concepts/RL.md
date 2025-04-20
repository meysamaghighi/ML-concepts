# 🧠 Reinforcement Learning (RL): A Complete Overview

---

## 🧩 What Is Reinforcement Learning?

Reinforcement Learning is a type of machine learning where **an agent** learns to make decisions by interacting with an **environment**, receiving **rewards** or **penalties** based on its actions.

The goal is to learn a **policy** $\pi(a|s)$ that maximizes the expected cumulative reward over time.

---

## 🎯 Formal Definition: Markov Decision Process (MDP)

RL problems are often formulated as **Markov Decision Processes (MDPs)**, defined by:

- $S$: Set of states  
- $A$: Set of actions  
- $P(s'|s, a)$: Transition probability  
- $R(s, a)$: Reward function  
- $\gamma$: Discount factor ($0 \leq \gamma \leq 1$)

The objective is to find a policy that maximizes:

```math
J(\pi) = \mathbb{E}_\pi \left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]
```

---

## 🔄 Core Elements

| Element | Description |
|--------|-------------|
| **Agent** | Learner or decision maker |
| **Environment** | The external system the agent interacts with |
| **State ($s$)** | The current situation |
| **Action ($a$)** | Possible choices the agent can make |
| **Reward ($r$)** | Feedback signal |
| **Policy ($\pi$)** | Strategy the agent follows |
| **Value Function ($V$ or $Q$)** | Expected long-term return |

---

## 🧭 RL Types and Categories

### By Model Use

| Type | Description | Example |
|------|-------------|---------|
| **Model-Free** | Learns policy/value from experience only | Q-learning, SARSA |
| **Model-Based** | Learns a model of the environment | Dyna-Q, MuZero |

### By Policy Behavior

| Type | Description | Example |
|------|-------------|---------|
| **On-Policy** | Learns using current policy | SARSA |
| **Off-Policy** | Learns from other policy’s experience | Q-learning, DQN |

### By Learning Focus

| Type | Description | Example |
|------|-------------|---------|
| **Value-Based** | Learn value functions | Q-learning |
| **Policy-Based** | Learn policy directly | REINFORCE |
| **Actor-Critic** | Learn both value and policy | A2C, DDPG, PPO |

---

## 🎰 Multi-Armed Bandits (MAB)

Special case of RL: no states, only actions (arms). Goal: maximize cumulative reward over time.

### Strategies to Balance Exploration–Exploitation

#### 1. ε-greedy

With probability $\varepsilon$ pick a random action, otherwise pick the best known:

```python
if random.random() < epsilon:
    action = random.choice(actions)
else:
    action = np.argmax(Q)
```

#### 2. Upper Confidence Bound (UCB)

```math
a_t = \arg\max_a \left( \hat{\mu}_a + c \sqrt{\frac{\ln t}{N(a)}} \right)
```

- $\hat{\mu}_a$: mean reward of arm $a$  
- $N(a)$: times arm $a$ has been chosen  
- $c$: exploration parameter

#### 3. Thompson Sampling

Bayesian approach that samples from posterior distributions.

```python
sample = np.random.beta(alpha[action], beta[action])
```

---

## 📐 Temporal Difference (TD) Learning

TD learning updates values based on other learned estimates.

---

## 🔑 Key Algorithms

### 1. SARSA (State–Action–Reward–State–Action)

**On-policy** learning:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
```

- Learns value of the policy currently being followed.

---

### 2. Q-learning

**Off-policy** learning: learns optimal policy regardless of current actions.

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
```

- More sample efficient than SARSA in many cases.

#### Python Code:
```python
Q[state][action] += alpha * (
    reward + gamma * max(Q[next_state]) - Q[state][action]
)
```

---

## 🧠 Deep Q-Networks (DQN)

DQN extends Q-learning using **deep neural networks** to handle large or continuous state spaces.

### Innovations

- **Experience Replay**: store transitions and sample them randomly
- **Target Network**: a separate network to stabilize learning

### DQN Loss Function:

```math
L(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a; \theta) \right)^2 \right]
```

### PyTorch-like Code:
```python
loss = F.mse_loss(
    policy_net(state)[action],
    reward + gamma * target_net(next_state).max(1)[0].detach()
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🛠 Model-Based RL (Brief)

Instead of just learning from experience, Model-Based RL:

1. **Estimates transition function** $P(s'|s,a)$ and reward $R(s,a)$.
2. **Plans ahead** (e.g., Monte Carlo Tree Search).

**Example**: MuZero from DeepMind (model-free training + model-based planning).

---

## 🚀 Applications of RL

| Industry | Use Case |
|----------|----------|
| **Retail** | Pricing, promotions, personalized recommendations |
| **NLP** | Chatbots, summarization with RL from Human Feedback (e.g., ChatGPT) |
| **Robotics** | Navigation, manipulation, locomotion |
| **Finance** | Portfolio optimization, trading bots |
| **Healthcare** | Adaptive treatment strategies |
| **Transportation** | Traffic light control, autonomous vehicles |
| **Games** | AlphaGo, OpenAI Five, Dota 2 |
| **Telecom** | Network optimization, handover policies |

---

## ⚖️ Comparison Table

| Algorithm | On/Off Policy | Model | Use Case | Complexity |
|-----------|---------------|--------|----------|------------|
| SARSA | On-policy | Model-free | Safer/real-time learning | Low |
| Q-learning | Off-policy | Model-free | Games, Planning | Low |
| DQN | Off-policy | Model-free + NN | Vision, Robotics, Games | High |
| Model-Based | Depends | Model-based | High sample efficiency | High |

---

## 🧪 Tips for Experimentation

- Use **replay buffers** and **target networks** for deep RL.
- Tune hyperparameters like $\varepsilon$, $\alpha$, $\gamma$.
- Normalize rewards for better training stability.
- Monitor training with tools like **TensorBoard** or **Weights & Biases**.

---

## 📚 Further Learning Resources

- **Book**: Sutton & Barto – *Reinforcement Learning: An Introduction*  
- **Course**: David Silver's RL Course (DeepMind)  
- **Libraries**: `OpenAI Gym`, `Stable-Baselines3`, `RLlib`  
- **Papers**: DQN (Mnih et al.), MuZero (Schrittwieser et al.)

---
