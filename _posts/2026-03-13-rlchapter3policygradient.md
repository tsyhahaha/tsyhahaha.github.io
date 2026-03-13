---
layout: post
title: "强化学习 Chapter 3 —— 跨越价值的苦海：策略梯度 (Policy Gradient) 与 REINFORCE"
date: 2026-03-13 14:25:20 +0800
author: tsyhahaha
categories: [RL, 笔记]
tags: [强化学习, RL, LLM, PPO, RLHF]
math: true
mermaid: true
---
# 强化学习 Chapter 3 —— 跨越价值的苦海：策略梯度 (Policy Gradient) 与 REINFORCE

> 在上一章中，基于价值 (Value-based) 的方法（如 DQN）展现了其在离散控制任务中的威力。然而，当我们将目光转向大语言模型（LLM）的生成任务时，Value-based 方法的局限性暴露无遗：面对动辄十万级规模的词表（Action Space）与极长的自回归序列，穷举 $\max_a Q(s, a)$ 引发了无法逾越的“维数灾难”。
> 
> 既然精确评估每个动作的价值在计算上不切实际，我们何不转换思路，直接参数化“动作生成概率”？

这一思想的转变，构成了现代大模型对齐算法（如 RLHF / RLVR 中的 PPO 与 GRPO）的核心基石：**策略梯度 (Policy Gradient)**。

---

## 1. 范式转移：从“估算价值”到“优化策略”

在 Value-based 算法中，我们在给定状态下评估所有候选动作的 $Q$ 值，进而推导出一个贪心策略。但在 LLM 生成文本的场景中，每一步的候选动作是整个词表（$ \mid V \mid \approx 10^5$），计算所有 $Q(s, a)$ 极其昂贵且效率低下。

**Policy-based 思想**提供了一条更直接的路径：直接使用神经网络 $\theta$（即 LLM 本身）参数化策略，输出在状态 $s$ 下的概率分布 $\pi_\theta(a \mid s)$。这与语言模型的自回归交叉熵预训练目标在形式上完美契合。

其核心优化逻辑直击本质：
1. **采样 (Rollout):** 模型根据当前策略 $\pi_\theta$ 与环境交互，生成一条轨迹（在 LLM 中即为根据 Prompt 生成 Response）。
2. **评估与反馈:** 如果该轨迹获得了高回报（High Reward），则在参数更新时推高轨迹中相应动作（Token）的生成概率；反之则压低。

---

## 2. 策略梯度定理 (Policy Gradient Theorem)

我们的优化目标 $J(\theta)$ 是寻找最大化**期望累计回报 (Expected Return)** 的策略参数 $\theta$：

$$

J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]

$$

直接对该期望求导面临一个数学障碍：轨迹回报的分布依赖于环境的未知动态转移矩阵 $P(s' \mid s,a)$，它是不可导的。幸运的是，借助**对数导数技巧 (Log-derivative trick)**：

$$

\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta

$$

我们能够巧妙地将关于分布的导数，转化为该分布下的期望（详见附录证明）。如此一来，环境的黑盒转移概率在求导过程中被完全消解，留下了著名的**策略梯度定理**：

$$

\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi}(s_t, a_t) \right]

$$


> **Post-Training 视角：加权自监督与 SFT 的统一**
>
> 审视此公式可以在 LLM 训练维度获得极佳的物理直觉。在 SFT（监督微调）阶段，损失函数为负对数似然 $-\log \pi_\theta(a^* \mid s)$，即我们强行推高极少数 Gold Token 的概率。
>
> 策略梯度的本质，则是**基于评估反馈的加权交叉熵优化**。在 RL 阶段，由于失去“标准答案”，动作的优劣由价值 $Q^{\pi}(s_t, a_t)$ 衡量。当 $Q > 0$ 时，模型受到正向激励，执行类似于 SFT 的梯度下降，强化这些 Token 的生成；当 $Q < 0$ 时，梯度反转，抑制不良偏好。SFT 与 RL 在损失函数的底层结构上实现了优雅的统一。

---

## 3. 朴素的起点：REINFORCE 算法

策略梯度定理中唯一的未知量是 $Q^{\pi}(s_t, a_t)$。作为最经典的实现，**REINFORCE** 算法采用了蒙特卡洛 (Monte Carlo) 估计法：让模型完整生成整条轨迹（Episode），获取真实的累计得分 $G_t$ 作为 $Q$ 值的无偏估计。

$$

G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}

$$

参数更新公式随即化简为：

$$

\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t

$$

在典型的 RLHF 设定中，这等价于：LLM 吐出完整的回复，被输入 Reward Model (RM) 打分，该标量奖励随后作为 $G_0$ 广播至生成该回复的每一个 Token 头上进行梯度更新。

---


## 4. 方差之魇与基线 (Baseline) 解救

然而，在工业级 LLM 对齐实践中，纯正的 REINFORCE 是难以直接收敛的。其作为纯蒙特卡洛方法的致命缺陷在于：**方差极高 (High Variance)**。

在大语言模型中，一条 Response 往往包含数百个 Token。如果仅仅因为个别标点符号的错误导致 RM 给出低分，REINFORCE 会一刀切地压低整段文本所有 Token 的概率——这种粗粒度的“信号广播”必然导致训练震荡。

### 引入 Baseline 降低方差

数学上，我们可以为回馈引入一个只依赖于状态的**基线 (Baseline) $b(s)$**。可以证明，减去基线**不会改变策略梯度的期望方向（保持无偏）**，但能极大地缩小梯度的方差：

$$

\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]

$$

在最佳实践中，$b(s)$ 通常被设定为状态价值函数 $V(s_t)$。此时，$G_t - V(s_t)$ 蜕变成了一个全新的度量：**优势函数 (Advantage Function, $A_t$)**。

$$

\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A(s_t, a_t)\right]

$$


> **Post-Training 视角：Advantage 抹平 Prompt 难度差异**
> 
> 在 LLM 场景中，不同的 Prompt (即初始状态 $s_0$) 难度差异巨大。写代码的满分可能只有 0.3，而打招呼的满分能到 0.9。如果没有 Baseline，模型可能会为了追求高绝对回报而拒绝回答困难问题，或者其参数更新被简单任务的梯度主导。
> 
> $V(s)$ 本质上是在预判“当前 Prompt 及已生成的上下文能带来多少预期收益”。将更新信号从绝对的 $G_t$ 转变为相对的 $A_t = G_t - V(s_t)$，模型优化的不再是“绝对表现”，而是“是否超出预期”。

### 对抗 Reward Hacking：KL 散度约束

除了高方差，RL 在无约束优化中极易出现策略退化。模型为了最大化标量奖励，会迅速利用 Reward Model 的漏洞（Reward Hacking），输出诸如冗长且毫无意义的高光词汇的“吉利话”或乱码。

在传统 RL 中，我们通过引入**熵正则化 (Entropy Regularization)** 来强制模型保持策略的随机性。而在 LLM Post-training（如 PPO）中，我们采用了更具针对性的结构化约束：**KL 散度惩罚**。

我们在 Reward 中动态惩罚新策略 $\pi_\theta$ 偏离参考模型 $\pi_{\text{ref}}$ 的距离：

$$

r_t^{\text{penalized}} = r_t - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}

$$

这不仅防止了模型“遗忘” SFT 阶段学习到的自然语言先验（Language Prior），也在本质上平滑了优化地形，从根本上缓解了策略退化。


## 附录

#### 1. 策略梯度定理 (Policy Gradient Theorem) 的推导过程

目标函数（轨迹的回报期望）：

$$

\begin{aligned}
J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \\
&= \int P(\tau \mid \theta) R(\tau) d\tau
\end{aligned}

$$

利用 Log-derivative 技巧求梯度：

$$

\begin{aligned}
\nabla_\theta J(\theta) &= \int \nabla_\theta P(\tau \mid \theta) R(\tau) d\tau \\
&= \int P(\tau \mid \theta) \nabla_\theta \log P(\tau \mid \theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau \mid \theta) R(\tau)]
\end{aligned}

$$

展开轨迹概率：

$$

P(\tau \mid \theta) = \rho_0(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t)

$$

对其取对数：

$$

\log P(\tau \mid \theta) = \log \rho_0(s_0) + \sum_{t=0}^T \log P(s_{t+1} \mid s_t, a_t) + \sum_{t=0}^T \log \pi_\theta(a_t \mid s_t)

$$

求梯度，由于前两项（初始状态概率和环境物理转移概率）与 $\theta$ 无关，恒为 0：

$$

\nabla_\theta \log P(\tau \mid \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)

$$

根据因果律（未来的预测不影响过去的收益），剥离积分并将 $R(\tau)$ 等效化为时刻 $t$ 的状态-动作长期期望回报 $Q^\pi(s_t, a_t)$，即得：

$$

\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^\pi(s_t, a_t) \right]

$$

#### 2. 基线 (Baseline) 的无偏性证明

证明减去仅与状态有关的基线 $b(s)$ 不改变期望梯度的方向：

$$

\begin{aligned}
\mathbb{E}_{a \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(a \mid s) b(s) ] &= \sum_{a} \pi_\theta(a \mid s) \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} b(s) \\
&= b(s) \nabla_\theta \sum_a \pi_\theta(a \mid s) \\
&= b(s) \nabla_\theta(1) = 0
\end{aligned}

$$

因此，Baseline 在不改变期望的前提下有效重塑了梯度的分布边界，大幅削减了方差。
