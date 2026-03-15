---
layout: post
title: "强化学习 Chapter 4 —— 学会边走边判：Actor-Critic 与 PPO"
date: 2026-03-15 16:01:14 +0800
author: tsyhahaha
categories: [RL, 笔记]
tags: [强化学习, RL, LLM, Actor-Critic, PPO, RLHF]
pin: false
toc: true
math: true
mermaid: true
---
# 强化学习 Chapter 4 —— 学会边走边判：Actor-Critic 与 PPO

> 上一章中，我们已经从策略梯度定理出发，得到了 REINFORCE 这一最原始的 Policy-based 算法。然而，理论上“正确”并不意味着工程上“可用”。REINFORCE 的问题在于：它必须等整条轨迹结束后才能更新，而且梯度方差极高。在动辄上百 Token 的 LLM 生成场景中，这样的学习信号既粗糙又不稳定。
>
> 于是，现代强化学习迈出了关键一步：**让策略网络负责行动，让价值网络负责评判**。这就是 **Actor-Critic**。而为了避免策略每次更新都迈得过猛、导致训练发散，工业界进一步引入了一个“带安全带的策略梯度”版本：**PPO (Proximal Policy Optimization)**。

如果说 REINFORCE 只是证明了“策略梯度这条路能走通”，那么 Actor-Critic 与 PPO 则真正构成了现代深度 RL 与 LLM Post-training 的主力工程范式。

---

## 1. 从 REINFORCE 到 Actor-Critic

REINFORCE 的更新形式非常纯粹：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t
$$

其中 $G_t$ 是从时刻 $t$ 往后的累计回报。问题也恰恰出在这里：$G_t$ 是一个高方差、长时延、纯采样的估计量。

它在工程上有三重痛点：

1. **必须等 Episode 结束才能更新。**  
   如果一条轨迹很长，学习反馈会被严重延迟。

2. **方差极高。**  
   同一个动作是否“好”，会被后续一长串随机事件污染。

3. **信用分配过于粗糙。**  
   尤其在 LLM 中，最后一句 RM 打分往往要反向归因到前面几十上百个 Token，极易造成训练震荡。

上一章我们通过引入 Baseline，把更新信号从 $G_t$ 变成了优势函数：

$$
A_t = G_t - V(s_t)
$$

但这只是第一步。Actor-Critic 架构真正的突破在于：**我们不再把 $V(s_t)$ 视作一个固定的数学对象，而是显式训练一个网络去逼近它。**

这意味着，策略更新不再依赖整条轨迹的“终局总结”，而是可以借助一个在线（online）学习的评论员，在每一步都获得更细粒度、更低方差的反馈。

---

## 2. Actor-Critic：一个负责行动，一个负责裁判

Actor-Critic 是对策略梯度最自然的工程化分工：

* **Actor（演员）**：参数化策略 $\pi_\theta(a \mid s)$，负责在状态 $s$ 下选择动作。
* **Critic（评论员）**：参数化价值函数 $V_\phi(s)$ 或 $Q_\phi(s,a)$，负责估计“当前局面未来能值多少钱”。

其核心思想极其朴素：**Actor 负责探索世界，Critic 负责告诉 Actor 哪些选择比预期更好。**

### 2.1 TD 在 Value-based 与 Policy-gradient 中的角色差异

时序差分（Temporal Difference, TD）并不是 Actor-Critic 独有的概念，但它在 **基于价值的 RL** 和 **策略梯度 RL** 中承担的职责并不一样。

在 Value-based 算法里，TD 是**主更新规则本身**。无论是 TD(0)、SARSA 还是 Q-Learning，本质上都在直接更新 $V$ 或 $Q$，试图让价值估计满足贝尔曼方程。例如 Q-Learning 的更新：

$$

Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_t+\gamma \max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]

$$

这里的 TD 误差直接承担了“学价值，并由价值导出策略”的全部职责。

而在策略梯度路线里，真正被优化的对象始终是策略 $\pi_\theta$。TD 不再直接产出策略，而是承担两个更间接的角色：

* 它为 **Critic** 提供价值学习的回归目标；
* 它为 **Actor** 提供一个低方差的 Advantage 近似。

因此，两条路线的根本区别在于：

* **Value-based RL**：TD 直接推动价值函数逼近最优解；
* **Actor-Critic**：TD 先帮助 Critic 学会评估，再由 Critic 反过来帮助 Actor 做策略提升。

### 2.2 Critic 的更新信号：从 Bellman Equation 到 TD Target

为什么 Critic 可以不等待整条轨迹结束，就提前学习“这个状态未来值多少钱”？答案来自贝尔曼方程。对于固定策略 $\pi$，状态价值满足：

$$

V^\pi(s_t)=\mathbb{E}\left[r_t+\gamma V^\pi(s_{t+1})\right]

$$

这意味着：**一个状态的长期价值，可以递归地拆成“眼前一步的奖励”加上“下一状态的折扣价值”。**

正因为有了这个递推结构，Critic 才可以把对未来的估计“借”回来更新当下，而不必等整条 Episode 结束。这种做法就是 **自举（Bootstrapping）**。在函数逼近情形下，我们用参数化网络 $V_\phi$ 去近似 $V^\pi$，于是得到 TD Target：

$$

y_t = r_t + \gamma V_\phi(s_{t+1})

$$

并令 Critic 通过回归这个目标来学习：

$$

\mathcal{L}_{\text{value}}(\phi) = \mathbb{E}\left[(V_\phi(s_t)-y_t)^2\right]

$$

如果把“当前估计”和“TD Target”做差，就得到单步时序差分误差：

$$

\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)

$$

因此，所谓“Critic 用 Bellman 自举不断修正自己的判断”，并不是一句口号，而是指：**Critic 每次都在用贝尔曼递推给出的局部一致性条件，逼迫自己的价值估计向更合理的方向收敛。**

### 2.3 Actor 的更新信号：从 TD Error 到 Advantage

TD 误差之所以能进一步服务于 Actor，是因为对真实价值函数而言，优势函数满足：

$$

A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
=
\mathbb{E}\left[r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t) \mid s_t,a_t\right]

$$

这说明：**在 Critic 足够准确时，TD 误差正是 Advantage 的一个一步近似。**

因此，在最朴素的 one-step Actor-Critic 中，我们直接令

$$

\hat A_t=\delta_t

$$

并据此更新 Actor：

$$

\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat A_t

$$

但现代 Actor-Critic 很少只停留在一步。更常见的做法，是把一串未来的 TD 误差继续加权累积，构造多步 Advantage 估计器，例如下一节要讲的 GAE。  所以更准确的表述是：**$\delta_t$ 是最基础的一步 Advantage 估计，而 GAE 则是由多步 TD 残差加权得到的更稳定版本。**

这就是 Actor-Critic 的本质：**Critic 用贝尔曼递推学习“局面值多少钱”，Actor 用 Critic 产生的 Advantage 信号学习“这个动作值不值得鼓励”。**

> **Post-Training 视角：Value Head 为什么如此关键？**
>
> 在 LLM 的 PPO 训练中，Actor 往往就是当前待对齐的语言模型本体，而 Critic 则常常是挂在同一主干网络上的一个 **Value Head**。  
> 它接收的状态不是传统控制任务里的物理坐标，而是当前的文本上下文 $s_t=(q,o_{<t})$；它输出的也不是“下一个 Token 是什么”，而是“从这个前缀继续生成下去，预期还能拿到多少总收益”。
>
> 这一步极其重要，因为它把原本只存在于终局的 RM 标量奖励，变成了一个沿着生成前缀不断传播的、可学习的价值场。

### 2.4 Actor-Critic 的灵魂：用 Advantage 做“相对评分”

上一小节已经说明，Critic 最终提供给 Actor 的，不是一个绝对回报，而是一个近似 Advantage 的更新信号。其意义在于：它给出了一个**相对评分标准**。

在 Actor-Critic 中，我们往往不会直接拟合昂贵的 $Q(s,a)$，而是通过 TD 结构近似得到一个优势估计量 $\hat A_t$，再用它指导策略更新。

这件事背后的哲学很重要：**Actor 不需要知道“这个动作最终拿了多少分”，它只需要知道“这个动作是否比当前状态下的平均预期更好”。**这也是为什么 Advantage 比原始 Return 更适合做训练信号：

* 它自动扣除了状态本身的难度；
* 它减少了不同 Prompt、不同轨迹之间的尺度差异；
* 它让策略优化更像“排序”而不是“硬记分”。

> **Prompt 难度归一化，本质上就是价值基线在起作用**
>
> 对同一个模型而言，“解释一个笑话”和“证明一道数学题”显然不是一个难度级别。  
> 如果直接用绝对回报更新，模型会天然偏向那些轻易拿高分的简单 Prompt。  
> Critic 的存在，等价于在每个状态上先问一句：“在这个上下文里，正常发挥大概能拿多少分？”  
> 策略真正优化的，不是绝对分数，而是**超额表现**。


---

## 3. 从单步 TD 到 GAE：在偏差与方差之间折中

如果直接使用单步 TD 误差 $\delta_t$ 作为 Advantage，方差虽然很低，但偏差会增大，因为它高度依赖 Critic 当前的估计质量；反之，如果退回到 Monte Carlo 的 $G_t-V(s_t)$，则偏差小，但方差又重新爆炸。

现代 Actor-Critic 的标准答案，是在两者之间取一个工程上更优雅的折中：**广义优势估计 (Generalized Advantage Estimation, GAE)**。

定义单步 TD 误差：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

则 GAE 形式为：

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{T-t-1}(\gamma \lambda)^l\delta_{t+l}
$$

这个公式非常容易让人误解：因为它显式依赖多个未来的 $\delta_{t+l}$，看上去似乎意味着 GAE 必须像 Monte Carlo 一样为每个时刻做“多步采样”。其实并不是。**GAE 需要的不是“为每个 $t$ 额外重启很多次采样”，而是“先拿到一批连续 rollout，再在每条 rollout 上做一次从后往前的后处理”。**

更准确地说：

* **TD**：只需要一条单步转移 $(s_t,a_t,r_t,s_{t+1})$ 就能更新；
* **MC**：采样单位是**完整 rollout**。先采多条完整 rollout，再用 rollout 中每个时刻的真实回报 $G_t$ 去估计期望；如果要估计 $V^\pi(s)$，本质上是在多个 rollout 的状态访问上做平均。
* **GAE**：同样从**一批 rollout** 出发，但它不是在 rollout 之间做另一层“平均”，而是在**同一条 rollout、同一个时刻 $t$** 上，把不同 horizon 的 $n$-step advantage estimator 通过 $\lambda$ 做加权混合。

这正是 GAE 在工业界如此常用的原因：它既不像单步 TD 那样只看一步、偏差偏大，也不必像 MC 那样完全依赖完整终局回报。

在实现上，工业界几乎不会按定义式逐项展开求和，而是采用一个等价的**反向递推（backward recursion）**。记 $d_t$ 为终止标记（终止则为 1，否则为 0），则：

$$
\delta_t = r_t + \gamma (1-d_t)V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$$
\hat A_t = \delta_t + \gamma \lambda (1-d_t)\hat A_{t+1}
$$

也就是说，采完一段 rollout 后，我们只需从尾到头扫一遍，就能在线性时间里得到整段轨迹的 GAE。

其中 $\lambda \in [0,1]$ 是一个极其关键的平衡旋钮：

* **$\lambda \to 0$**：更接近单步 TD，低方差、高偏差。
* **$\lambda \to 1$**：更接近 Monte Carlo，低偏差、高方差。

在 LLM PPO / RLHF 中，这段 rollout 往往对应“整条回答生成完成后”的整段 token 序列。也就是说，**LLM 里 GAE 的常见计算方式不是每个 token 单独采样未来，而是先采一批完整 responses，再在每条 response 内沿 token 维度做一次 backward pass，最后对 batch 中所有 token 的 loss 做平均。**

---

## 4. PPO：跨越现实应用与理论之间的鸿沟

PPO 全称 **Proximal Policy Optimization**。如果说 Actor-Critic 解决的是“谁来评估、谁来更新”的架构问题，那么 PPO 解决的则是另一个更工程化、也更棘手的问题：

**策略梯度在理论上是 on-policy 的，但真实系统又必须高效复用样本。**

这正是 PPO 诞生的背景。它不是凭空发明了一个新目标，而是试图修补**理论要求的严格 on-policy 更新**与**现实训练中 rollout 成本极高**之间的鸿沟。

### 4.1 从 on-policy 困境到重要性采样

回忆策略梯度的基本形式：我们真正想优化的是当前策略 $\pi_\theta$ 下的期望回报，因此梯度写成

$$

\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat A_t
\right]

$$

这条公式隐含了一个非常强的理论要求：**采样分布必须和当前策略一致。**  也就是说，只要参数更新了一步，严格意义上我们就应该重新用新的 $\pi_\theta$ 再采一批 rollout。问题在于，现实系统根本承受不起这种“更新一步，重采一次”的成本。尤其在 PPO / RLHF 场景中，一条 response 往往很长，RM 打分、value 计算和反向传播都很贵。工程上更自然的做法是：

* 先冻结当前策略，记为 $\pi_{\theta_{\text{old}}}$；
* 用它采一大批 rollout；
* 在同一批数据上做多轮 minibatch 更新。

但这样一来，理论与现实之间的鸿沟就出现了：**我们手里的数据来自 $\pi_{\theta_{\text{old}}}$，而想优化的却是新的 $\pi_\theta$。**

重要性采样正是用来修补这道鸿沟的。对于任意函数 $f(s_t,a_t)$，有

$$

\mathbb{E}_{a_t\sim\pi_\theta}[f(s_t,a_t)]
=
\mathbb{E}_{a_t\sim\pi_{\theta_{\text{old}}}}
\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}f(s_t,a_t)
\right]

$$

于是我们定义动作概率比：

$$

r_t(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}

$$

它的物理直觉非常明确：**同一个 old policy 样本里的动作 $a_t$，在新策略下相对被放大或缩小了多少概率。**

### 4.2 surrogate objective

严格地说，如果要把整个目标从 $\pi_\theta$ 改写到 $\pi_{\theta_{\text{old}}}$，应该对整条轨迹做重要性采样：

$$

\mathbb{E}_{\tau\sim\pi_\theta}[F(\tau)]
=
\mathbb{E}_{\tau\sim\pi_{\theta_{\text{old}}}}
\left[
\frac{P_\theta(\tau)}{P_{\theta_{\text{old}}}(\tau)}F(\tau)
\right]

$$

其中轨迹比率是每一步动作概率比的连乘：

$$

\frac{P_\theta(\tau)}{P_{\theta_{\text{old}}}(\tau)}
=
\prod_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}

$$

但这在长序列里方差极高，几乎不可用。于是 PPO / TRPO 的做法是：**只在 old policy 附近做局部近似，把状态分布和 Advantage 估计固定在 $\pi_{\theta_{\text{old}}}$ 上，只保留每一步的 action-level ratio。**

于是得到 PPO 的 surrogate objective：

$$

\mathcal{L}^{\text{PG}}(\theta)
=
\mathbb{E}_{s_t,a_t\sim\pi_{\theta_{\text{old}}}}
\left[
r_t(\theta)\hat A_t
\right]

$$

之所以叫 **surrogate objective**，正是因为它已经不是原始的 $J(\theta)$，而是一个在 $\pi_{\theta_{\text{old}}}$ 附近近似原目标、但更可计算也更低方差的替代目标。  它解决了“如何用旧策略采来的数据近似优化新策略”这个现实问题，但同时也埋下了新的隐患：**如果新旧策略偏得太远，这个近似就会迅速失真。**

### 4.3 clipping：给 surrogate objective 戴上安全带

PPO 的核心发明，是继续在 surrogate objective 上加一道“安全带”，构造出 clipped objective：

$$

\mathcal{L}^{\text{CLIP}}(\theta)=
\mathbb{E}\left[
\min \left(
r_t(\theta)\hat A_t,\;
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]

$$

这里的 $\epsilon$ 通常是一个较小的常数（如 0.1 或 0.2），它定义了一个“允许偏移区间”。这个目标函数的物理意义非常漂亮：

* 当策略更新还在安全范围内时，PPO 与普通策略梯度几乎一致；
* 一旦 $r_t(\theta)$ 超过允许范围，目标增益就会被截断；
* 这相当于告诉优化器：**可以变好，但别变得太快。**

更具体地说：

* 若 $\hat A_t>0$，说明该动作值得鼓励，此时如果 $r_t(\theta)>1+\epsilon$，PPO 不再继续奖励你把概率推得更高；
* 若 $\hat A_t<0$，说明该动作应该被压制，此时如果 $r_t(\theta)<1-\epsilon$，PPO 也不再鼓励你无限制地把它打到地板。

于是，PPO 形成了一条完整的逻辑链：

* **理论要求**：策略梯度必须 on-policy；
* **工程现实**：rollout 太贵，必须复用 old policy 数据；
* **importance sampling**：把“新策略目标”改写到“旧策略样本”上；
* **surrogate objective**：得到一个局部可优化的近似目标；
* **clipping**：限制近似目标在偏移过大时失真。

> **PPO 并没有把 RL 变简单，它只是把“不可控的大步前进”削成了“可控的小步快跑”**
>
> 这也是 PPO 在工业界如此成功的根本原因。  它不是最优控制理论意义上最优雅的算法，却是深度神经网络语境下最稳健、最能落地的一种折中：  **允许重复利用样本，但禁止策略在一次更新周期里失控。**

### 4.4 PPO 在 LLM 中如何落地

一个常见误解是：PPO 是某种“替代 Actor-Critic 的新算法”。事实恰恰相反，**PPO 本质上仍然是一种 Actor-Critic 的稳定优化器**。它保留了全部核心组件：

* **Actor**：参数化策略 $\pi_\theta$
* **Critic**：估计价值 $V_\phi(s)$
* **Advantage Estimator**：通常采用 GAE
* **Policy Update Rule**：把原始策略梯度替换成 clipped objective

因此，更准确地说：

**Actor-Critic 解决的是“谁来评估、谁来更新”的架构问题；PPO 解决的是“如何在复用样本时仍然安全更新”的优化问题。**

当 PPO 被搬到大语言模型训练中时，变量只是换了一个外壳，底层逻辑几乎没有变化。设输入 Prompt 为 $q$，模型输出为 $o=(o_1,\dots,o_T)$，则状态写成：

$$

s_t=(q,o_{<t})

$$

PPO 在 LLM RLHF 中的典型流水线如下：

1. **采样 Prompt**  
   从数据分布中取一批问题 $q \sim P(Q)$。

2. **冻结旧策略并生成 Response**  
   复制当前模型为 $\pi_{\theta_{\text{old}}}$，用它生成完整回答 $o$。

3. **构造 Token-level Reward**  
   终局由 Reward Model 给出句子级打分，同时叠加每一步相对参考模型 $\pi_{\text{ref}}$ 的 KL 惩罚：
   $$

 r_t^{\text{penalized}}
 =
 r_t
 -
 \beta \log \frac{\pi_\theta(o_t \mid q,o_{<t})}{\pi_{\text{ref}}(o_t \mid q,o_{<t})}
 $$

   
4. **训练 Critic 并计算 GAE**  
   用 Value Head 预测每个前缀状态的价值，再据此构造 $\hat A_t$。

5. **执行多轮 minibatch PPO 更新**  
   对同一批旧轨迹重复若干次优化 clipped objective 与 value loss。

6. **刷新 old policy，进入下一轮 rollout**  
   更新后的模型成为新的采样策略，再进行下一轮 on-policy 训练。

写成 LLM 语境下的 PPO 目标，可以表示为：

$$

\mathcal{L}^{\text{CLIP}}(\theta)=
\mathbb{E}_{q,o\sim \pi_{\theta_{\text{old}}}}
\left[
\frac{1}{ \mid o \mid }
\sum_{t=1}^{ \mid o \mid }
\min \left(
r_t(\theta)\hat A_t,\;
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]

$$

其中

$$

r_t(\theta)=
\frac{\pi_\theta(o_t \mid q,o_{<t})}
{\pi_{\theta_{\text{old}}}(o_t \mid q,o_{<t})}

$$

> **务必区分两种“参考策略”**
>
> 在 LLM PPO 文献里，最容易混淆的其实是两个不同角色：
>
> * $\pi_{\theta_{\text{old}}}$：用于 PPO 重要性采样的**旧策略快照**；
> * $\pi_{\text{ref}}$：用于 KL 正则的**参考模型**，通常是 SFT 模型。
>
> 前者服务于“如何安全更新”，后者服务于“不要偏离人类语言分布太远”。  
> 一个是优化器视角的参照系，一个是对齐约束视角的护栏。
>
> PPO 在 LLM 中格外有用，主要因为：
>
> * **response 很长，样本极贵**：必须在同一批 old policy rollout 上做多轮更新；
> * **reward 稀疏且高噪声**：必须依赖 Critic 和 GAE 把终局反馈传播到每个 token；
> * **模型表达能力太强**：如果没有 clip 与 KL 的双重约束，reward hacking 会来得非常快。

---

**References:**

* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
* [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)

## 附录

#### 1. TD 误差近似 Advantage

对真实价值函数 $V^\pi$ 而言，有：

$$
Q^\pi(s_t,a_t)=r_t+\gamma \mathbb{E}[V^\pi(s_{t+1})]
$$

因此

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
=
\mathbb{E}[r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)]
$$

也就是说，在 Critic 足够准确时，单步 TD 误差

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

正是 Advantage 的一个低方差近似。

#### 2. GAE 的理论推导

先定义一个 $n$ 步 advantage 估计量：

$$
\hat A_t^{(n)}
=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)
$$

它表示：向前看 $n$ 步真实奖励，再用第 $n$ 步后的 Critic 估值来补尾。

另一方面，单步 TD 误差为：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

将未来连续 $n$ 个 TD 误差按折扣因子加总：

$$
\sum_{l=0}^{n-1}\gamma^l \delta_{t+l}
=
\sum_{l=0}^{n-1}\gamma^l \left(r_{t+l}+\gamma V_\phi(s_{t+l+1})-V_\phi(s_{t+l})\right)
$$

展开后可以看到中间的价值项会消去：

$$
\sum_{l=0}^{n-1}\gamma^l \delta_{t+l}
=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)
=
\hat A_t^{(n)}
$$

也就是说，**$n$ 步 Advantage 可以等价写成一串 TD 误差的折扣和**。

接下来引入 $\lambda$，对所有步长的 $n$ 步估计做指数加权混合：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}
$$

把 $\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$ 代入：

$$
\hat A_t^{\mathrm{GAE}}
=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
$$

交换求和顺序：

$$
\hat A_t^{\mathrm{GAE}}
=
\sum_{l=0}^{\infty}\gamma^l\delta_{t+l}(1-\lambda)\sum_{n=l+1}^{\infty}\lambda^{n-1}
$$

而

$$
(1-\lambda)\sum_{n=l+1}^{\infty}\lambda^{n-1}=\lambda^l
$$

于是得到：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

这就是正文中的 GAE 公式。有限长轨迹时，只需把无穷上界截断到 $T-t-1$ 即可。
