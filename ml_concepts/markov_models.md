### 📊 Markov Models

| Model                         | Controlled | Observability | Temporal? | Use Case                            | Python Library        |
|------------------------------|------------|---------------|-----------|-------------------------------------------|---------------------------|
| **Markov Chain (MC)**        | ❌ No       | ✅ Fully      | ✅ Yes    | Sequence generation, simulation           | `numpy`, custom code      |
| **Hidden Markov Model (HMM)**| ❌ No       | ❌ Partial    | ✅ Yes    | POS tagging, speech recognition           | `hmmlearn`, `pomegranate` |
| **Markov Decision Process (MDP)** | ✅ Yes  | ✅ Fully      | ✅ Yes    | Reinforcement Learning (RL)               | `MDPtoolbox`, `OpenAI Gym`|
| **Partially Observable MDP (POMDP)** | ✅ Yes | ❌ Partial   | ✅ Yes    | Robotics, planning under uncertainty      | `pomdp_py`, `POMDPs.jl` (Julia) |
| **Input-Output HMM (IOHMM)** | ✅ Yes      | ❌ Partial    | ✅ Yes    | Time series with exogenous inputs         | `pomegranate`             |
| **Dynamic Bayesian Network (DBN)** | ✅/❌   | ✅/❌ Mixed   | ✅ Yes    | Multivariate temporal modeling            | `pgmpy`, `pomegranate`    |
| **Markov Network (MRF)**     | ❌ No       | ✅/❌ Mixed    | ❌ No     | Image denoising, spatial dependencies     | `pgmpy`, `OpenGM`         |
| **Conditional Random Field (CRF)** | ✅ Yes  | ✅/❌ Mixed    | ✅ Yes    | Named Entity Recognition, sequence labeling| `sklearn-crfsuite`, `pystruct` |

---

### ✅ Legend

- **Controlled**: Whether the agent takes actions to influence transitions.
- **Observability**: Whether states are fully known or must be inferred.
- **Temporal**: Does the model describe transitions over time?
