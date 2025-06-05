<<<<<<< HEAD
# Reinforcement Learning Rumble

In this repository, we implement various reinforcement learning algorithms from scratch, including Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Variational Free Energy (VFE).

#### Goals

1. To become more proficient at translating research papers into code.
2. To gain an in-depth understanding of both the theory and practical implementation.
3. To become familiar with the tricks and techniques required to make RL work in practice.

## Table of Contents

- [Reinforcement Learning Rumble](#reinforcement-learning-rumble)
      - [Goals](#goals)
  - [Table of Contents](#table-of-contents)
  - [General introduction to Reinforcement Learning](#general-introduction-to-reinforcement-learning)
      - [What is Reinforcement Learning?](#what-is-reinforcement-learning)
      - [How to implement it in code?](#how-to-implement-it-in-code)
  - [PPO](#ppo)
      - [What is PPO?](#what-is-ppo)
      - [How do you connect theory to code?](#how-do-you-connect-theory-to-code)
      - [What are the optimization tricks?](#what-are-the-optimization-tricks)

## General introduction to Reinforcement Learning

#### What is Reinforcement Learning?

#### How to implement it in code?

## PPO 

#### What is PPO? 

$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}_{t} \left[ \log \pi(a_t | s_t) \cdot \hat{A}_t \right]
$$
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} - \beta \cdot \mathbb{E}_{t}[\mathcal{H}(\pi(\cdot | s_t))]
$$
$$
\mathcal{H}(\mathcal{N}(\mu, \Sigma)) = \frac{1}{2} \ln \left[ (2\pi e)^d \cdot \det(\Sigma) \right]
$$
In particular we have 
$$
\mathcal{H} = \frac{d}{2} \left( \ln(2\pi e) + \frac{1}{d} \sum_{i=1}^d \ln \sigma_i^2 \right)
$$

#### How do you connect theory to code?

1. When defining the advantage function, we fix the value of the Value function at the beginning of each episode to avoid unstable moving average computation.
2. Normalize the advantage function to have zero mean and unit variance

#### What are the optimization tricks?

- Learning rate annealing
- Mini batch update
- Entropy regularization
  - Prevent convergence suboptimal policy
  - Encourage the policy to stay stochastic 
  - High entropy = high exploration
- [Gradient clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
  - Exploding gradient can lead to unstable training 
- [Approximate KL divergence](http://joschu.net/blog/kl-approx.html)
  - Use the ratio of the new and old policy to approximate the KL divergence
  - The ratio is clipped to prevent large updates to the policy
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438)
  - Use a weighted sum of the advantages over multiple time steps to reduce variance
  - The weights are determined by a hyperparameter lambda
  - GAE is a trade-off between bias and variance
=======
#  🤖 RL-Rumble

A personal project to implement and benchmark three foundational reinforcement learning algorithms from scratch:

- **Proximal Policy Optimization (PPO)**
- **Variational Free Energy (VFE)**
- **Soft Actor-Critic (SAC)**

The goal is to understand their core mechanics, compare their performance, and build a unified framework for evaluating RL methods across diverse environments.

---

## ✅ To-Do List

### 🏋️ PPO (Proximal Policy Optimization)
- [ ] Improve performance metrics (e.g., average reward, stability)
- [ ] Evaluate across multiple seeds and environments

### 🧠 VFE – Variational Free Energy / Active Inference
- [ ] Design generative and recognition models
- [ ] Implement inference and action selection loop
- [ ] Validate behavior in simple discrete environments
- [ ] Extend to continuous action environments (optional/advanced)

### 🔥 SAC – Soft Actor-Critic
- [ ] Build full SAC pipeline (actor, twin critics, entropy regularization)
- [ ] Integrate replay buffer and soft target updates
- [ ] Benchmark in continuous control tasks (e.g. Pendulum, HalfCheetah)
---

## 📊 Coming Soon
- Benchmark results and learning curves
- Agent behavior visualizations
- A unified interface for evaluating algorithm performance

Stay tuned as each agent comes online!
>>>>>>> c092fd7294c77c911aef7e82aca3ec81826deb50
