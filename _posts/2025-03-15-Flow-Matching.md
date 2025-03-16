---
title: Flow Matching 理论学习
author: liliazhen669
date: 2025-03-15 17:00:00 +0800
categories: [Learning, Diffusion Model]
tags: [flow matching] # TAG names should always be lowercas
render_with_liquid: false
usemath: latex
---


> 论文链接：*[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)*

前几天看到了一篇Latent Bridge Diffusion的工作，感觉在Relighting方面的效果很惊艳，追溯了一下原理后发现又来到了Flow Matching这个概念，由于之前了解Stable Diffusion 3的时候接触过这个概念，但没有深入了解，现特意来学习Flow Matching

# Flow Matching

概念总结：Flow Matching实际上是一种Continuous Normalizing Flows的方法，该方法通过学习与概率路径相关的向量场(Vector Field)来训练模型，并使用常微分求解器（ODE Solver）来生成新样本。

Stable Diffusion 3中使用了Flow Matching，使用Flow Matching可以提高训练的稳定性。实际上，Diffusion Model是Flow Matching的一个特例，此外使用最优传输(Optimal Transport)技术构建概率路径可以进一步加快训练速度，并提高模型的泛化能力。

## Normalizing Flow

Normalizing Flows(NFs)是一种可逆的概率密度变换方法，其通过一系列可逆的变换函数来逐步将一个简单分布（如标准高斯分布）转换成一个复杂的概率分布。那么在 Normalizing Flow 训练完成后，就可以通过从简单概率分布中进行采样，然后通过逆变换得到复杂的目标分布中的样本

从这个角度看，Normalizing Flow 和 Diffusion Model 是有一些相通的，其做法的对比如下表所示。从表中可以看到，两者大致的过程是非常类似的，尽管依然有些地方不一样，但这两者应该可以通过一定的方法得到一个比较统一的表示。

从这个角度看，Normalizing Flow 和 Diffusion Model 是有一些相通的，其做法的对比如下表所示。从表中可以看到，两者大致的过程是非常类似的，尽管依然有些地方不一样，但这两者应该可以通过一定的方法得到一个比较统一的表示。

从这个角度看，Normalizing Flow 和 Diffusion Model 是有一些相通的，其做法的对比如下表所示。从表中可以看到，两者大致的过程是非常类似的，尽管依然有些地方不一样，但这两者应该可以通过一定的方法得到一个比较统一的表示。

| 模型             | 前向过程                                                           | 反向过程                                                               |
| ---------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Normalizing Flow | 通过显式的可学习变换将样本分布变换为标准高斯分布                   | 从标准高斯分布采样，并通过上述变换的逆变换得到生成的样本               |
| Diffusion Model  | 通过不可学习的 schedule 对样本进行加噪，多次加噪变换为标准高斯分布 | 从标准高斯分布采样，通过模型隐式地学习反向过程的噪声，去噪得到生成样本 |

## Continuous Normalizing Flow

简单来说，Continuous Normalizing Flow（CNF）是 Normalizing Flow 的连续形式，CNF通常用常微分方程（ODE）来表示：

$$
\frac{\mathrm{d}\mathbf{z}_t}{\mathrm{d}t}=v(\mathbf{z}_t,t) 
$$

其中 $t \in [0,1]$,
$\mathbf{z}_t$ 称为**Flow Map**,或者**Transport Map**,可以理解为时刻 $t$ 下的数据点，$v(\mathbf{z}_t,t)$ 是一个向量场(Vector Field)，定义了每个数据点在状态空间中时刻 $t$ 下的变化大小与方向，通常由神经网络来学习。当神经网络完成了对向量场 $v(\mathbf{z}_t,t)$ 的学习后，就可以用如下的欧拉方法来求解：

$$ \mathbf{z}_{t+\Delta t}=\mathbf{z}_t+\Delta t\cdot v(\mathbf{z}_t,t) $$

这意味着，给定一个初始概率分布（通常是标准高斯分布），可以通过学习向量场来学习数据的变换过程，从标准高斯分布采样，然后通过上述迭代过程得到目标分布中的一个近似解，完成生成的过程，从而实现从简单概率分布得到目标概率分布

## Continuity Equation

由于概率密度函数的性质确保了在全体分布上的积分为1，这反映了概率的总和是固定的，即概率是守恒的。在CNFs中，可以将这个性质与流体力学中的连续性方程类比，从而得到概率密度的连续性方程，这个方程可以写成：

$$
\frac{\partial p_t(\mathbf{x})}{\partial t}+\mathrm{div}(p_t(\mathbf{x})v_t(\mathbf{x}))=0
$$

其中 $p_t\left(\mathbf{x}\right)$
是时刻 $t$ 对应的概率密度函数、$v_t(\mathbf{x})$ 是与 $p_t(\mathbf{x})$ 关联的向量场。这个式子是向量场
$v_t(\mathbf{x})$ 能够产生概率密度路径  $p_t(\mathbf{x})$ 的充分必要条件，在后续的推导中会用这个式子作为一个约束来使用。

## Flow Matching

Flow Matching 的训练目标和 Score Matching 是比较类似的，学习的目标就是通过学习拟合一个向量场 $u_t$，使得能够得到对分布进行变换的概率路径 $p_t$，也就是下边这个公式：

$$
\mathcal{L}_\mathrm{FM}(\theta)=\mathbb{E}_{t,p_t(x)}||v_t(x)-u_t(x)||^2
$$

其中 $\theta$ 是模型的可训练参数，$t$ 在 0 到 1 之间均匀分布，$x\sim p_t(x)$ 是概率路径，$v_t(x)$ 是由模型表示的向量场。从上式中可以知道，Flow Matching 目标的核心是最小化这个损失函数，使得它预测的向量场 $v_t(x)$
尽可能接近于实际的向量场 $u_t \left ( x \right )$ 
，从而能够准确地生成目标概率路径 $p_t$。尽管训练目标很直观，但由于不知道如何确定合适的 $p_t(x)$ 和
$u_t \left ( x \right )$，因此无法直接使用，为了解决这个问题，在原论文中提出并证明了三条定理，以及提出了Conditional Flow Matching来解决这个问题

**定理一：**给定向量场 $u_t(x|x_1)$，其能够生成条件概率路径 
$p_t \left( x|x_1 \right )$，那么对于任意分布 $q \left(x_1 \right)$，满足某一特定形式（后文会给出）的边缘向量场 $u_t \left( x \right)$就能生成对应的边缘概率路径 
$p_t \left( x \right)$。

**证明：**首先，对于边缘概率路径 $p_t(x)$，有以下等式：

$$
p_t(x)=\int p_t(x|x_1)q(x_1)\mathrm{d}x_1
$$

进而可以推导：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}t}p_t(x)&=\frac{\mathrm{d}}{\mathrm{d}t}\int p_t(x|x_1)q(x_1)\mathrm{d}x_1 \\
&=\int\frac{\mathrm{d}}{\mathrm{d}t}p_t(x|x_1)q(x_1)\mathrm{d}x_1&\mathrm{Leibniz~integral~rule} \\
&=-\int\mathrm{div}(u_t(x|x_1)p_t(x|x_1))q(x_1)\mathrm{d}x_1&\mathrm{Continuity~equation} \\
&=-\mathrm{div}\left(\int u_t(x|x_1)p_t(x|x_1)q(x_1)\mathrm{d}x_1\right)&\mathrm{Leibniz~integral~rule} \\
\end{aligned}
$$

又根据连续性方程：

$$
\frac{\mathrm{d}}{\mathrm{d}t}p_t(x)=-\mathrm{div}\left(u_t(x)p_t(x)\right)
$$

两个式子联立得到 $u_t(x)$ 需要满足以下形式：

$$
u_t(x)=\int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}\mathrm{d}x_1
$$

也就是说，只要 
$p_t(x)$ 满足上边等式中的形式，就可以用 $u(x|x_1)$
和 $p(x|x_1)$ 取代 
$u(x)$ 和 $p(x)$。

虽然基于上述过程已经推导出了 $u_t(x)$ 的形式，但上述的积分依然不容易求解。因此作者给出了一种更容易求解的形式（如下所示），

$$
\mathcal{L}_\mathrm{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}||v_t(x)-u_t(x|x_1)||^2
$$

并且证明了下面这个损失函数与原本损失函数的等价性，即作者证明了 $\mathcal{L}_{CFM}$ 
和 $\mathcal{L}_{FM}$ 的等价性，也就是说优化 
$\mathcal{L}_{CFM}$ 等价于优化 $\mathcal{L}_{FM}$，可以用如下定理来描述：

$\theta$

**定理二：**假定对于所有 $x\in\mathbb{R}^d$ 且 $t\in[0,1]$ 
都有 $p_t(x)>0$，那么 $\mathcal{L}_{CFM}$ 
和 $\mathcal{L}_\mathrm{FM}$ 
相差一个与 $\theta$ 无关的常数，即有 $\nabla_\theta\mathcal{L}_\mathrm{FM}(\theta)=\nabla_\theta\mathcal{L}_\mathrm{CFM}(\theta)$。

**证明：**首先把两个二次项都展开，然后证明右侧是相等的。注意，虽然右侧都有 
$\left\Vert v_t(x)\right\Vert^2$ 这一项，但由于 
$\mathbb{E}$ 的下标不一样，所以不能直接认为两者相等。

$$
\begin{align}
\left\Vert v_t(x)-u_t(x)\right\Vert^2&=\left\Vert v_t(x)\right\Vert^2-2\left\langle v_t(x),u_t(x)\right\rangle+\left\Vert u_t(x)\right\Vert^2 \\
\left\Vert v_t(x)-u_t(x|x_1)\right\Vert^2&=\left\Vert v_t(x)\right\Vert^2-2\left\langle v_t(x),u_t(x|x_1)\right\rangle+\left\Vert u_t(x|x_1)\right\Vert^2
\end{align}
$$

由于 $u_t$ 相当于 GroudTruth，和 $\theta$ 无关，所以不产生梯度，在计算时可以直接略去最后一项。分别证明前两项相等：

$$
\begin{aligned}
\mathbb{E}_{p_t(x)}\left\Vert v_t(x)\right\Vert^2&=\int\left\Vert v_t(x)\right\Vert^2p_t(x)\mathrm{d}x \\
&=\int\left\Vert v_t(x)\right\Vert^2p_t(x|x_1)q(x_1)\mathrm{d}x_1\mathrm{d}x \\
&=\mathbb{E}_{q(x_1),p_t(x|x_1)}\left\Vert v_t(x)\right\Vert^2
\end{aligned}
$$

$$
\begin{aligned}
\mathbb{E}_{p_t(x)}&=\int\left\langle v_t(x),\frac{\int u_t(x|x_1)p_t(x|x_1)q(x_1)\mathrm{d}x_1}{p_t(x)}\right\rangle p_t(x)\mathrm{d}x \\
&=\int\left\langle v_t(x),\int u_t(x|x_1)p_t(x|x_1)q(x_1)\mathrm{d}x_1\right\rangle\mathrm{d}x \\
&=\int\left\langle v_t(x),u_t(x|x_1)\right\rangle p_t(x|x_1)q(x_1)\mathrm{d}x_1\mathrm{d}x \\
&=\mathbb{E}_{q(x_1),p_t(x|x_1)}\left\langle v_t(x),u_t(x|x_1)\right\rangle
\end{aligned}
$$

如此即证明了上述的定理。这样，我们的训练就不再依赖于一个抽象的边缘向量场，而是依赖于 $x_1$ 的条件向量场。这样我们就可以利用一定的训练数据对模型进行训练。

上面证明了条件概率路径和条件向量场可以等价于边缘概率路径和边缘向量场，并且用 CFM 的方式进行训练和 Flow Matching 的效果是相同的。但现在 $u_t(x|x_1)$ 的形式依然是不知道的，因此我们需要进一步定义具体的条件概率路径的形式。就像 DDPM，我们需要定义具体的前向过程，才能基于这个过程进行训练。

作者给出的条件概率路径的形式为：

$$
p_t(x|x_1)=\mathcal{N}(x|\mu_t(x_1),\sigma_t(x_1)^2I)
$$

其中 $\mu$ 是和时间有关的高斯分布均值，$\sigma$ 是和时间有关的高斯分布方差。并且为了使这个条件概率路径有比较良好的性质，作者设定在 $t=0$ 时 $p(x)$
为标准高斯分布 $\mathcal{N}(x|0,I)$
，也就是 $\mu_0(x_1)=0$、$\sigma_0(x_1)=1$；同时希望条件概率路径最终能够生成目标样本，所以当 $t=1$ 时高斯分布的均值和方差 $\mu_1(x_1)=x_1$、$\sigma_1(x_1)=\sigma_\min$，其中 $\sigma_\min$ 时一个足够小的数。

同时，作者将 Flow 定义为以下形式：

$$
\psi_t(x)=\sigma_t(x_1)x+\mu_t(x_1)
$$

其中 $x\sim\mathcal{N}(0,I)$ 服从标准高斯分布，根据上文所述的 CNF 的 ODE 表示，有：

$$
\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x)=u_t(\psi_t(x)|x_1)
$$

这样我们就可以将损失函数 $\mathcal{L}_\mathrm{CFM}$ 的形式变为如下形式：

$$
\mathcal{L}_\mathrm{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p(x_0)}\left\Vert v_t(\psi_t(x_0))-\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0)\right\Vert^2
$$

在上边的式子里，$\psi_t$ 的形式是已知的，并且 $x_0\sim\mathcal{N}(0,I)$，所以上边的式子是可以求解的，是实用、可以实现的损失函数。同时，也可以得到条件向量场的形式，即：

**定理三：**令 $p_t(x|x_1)$ 
是上述的高斯概率路径，$\psi_t$ 是上述的 flow map，那么 $\psi_t(x)$ 对应于唯一的向量场 
$u_t(x|x_1)$，且形式为：

$$
u(x|x_1)=\frac{\sigma'_t(x_1)}{\sigma_t(x_1)}(x-\mu(x_1))+\mu'_t(x_1)
$$

**证明：**由于 $\psi_t$ 可逆，令 $x=\psi^{-1}(y)$，则可以写出：

$$
\psi^{-1}(y)=\frac{y-\mu_t(x_1)}{\sigma_t(x_1)}
$$

同时对 $\psi_t$ 求导得到：

$$
\psi'_t(x)=\sigma'_t(x_1)x+\mu'_t(x_1)
$$

根据 ODE，推导得到：

$$
\begin{aligned}
u_t(y|x_1)&=\psi'_t(x)=\psi'_t(\psi_t^{-1}(y))=\psi'_t(\sigma'_t(x_1)y+\mu'_t(x_1)) \\
&=\frac{\sigma'_t(x_1)}{\sigma_t(x_1)}(y-\mu_t(x_1))+\mu'_t(x_1)
\end{aligned}
$$

## 讨论

Flow Matching 定义了一种特定形式的高斯概率路径，当选择不同的均值和方差时，有几种特殊的情况：

- Variance Exploding: 
$p_t(x)=\mathcal{N}(x|x_1,\sigma_{1-t}^2I)$，其中 $\mu_t(x_1)=x_1$、$\sigma_t(x_1)=\sigma_{1-t}$，并且 $\sigma_t$ 是递增函数，$\sigma_0=0$、$\sigma_1\gg1$。这种过程能够使模型生成数据时探索范围更广的空间，有助于生成多样的样本。
- Variance Preserving:
$p_t(x|x_1)=\mathcal{N}(x|\alpha_{1-t}x_1,(1-\alpha_{1-t}^2)I) $，其中 $\mu_t(x_1)=\alpha_{1-t}x_1$、$\sigma_t(x_1)=\sqrt{1-\alpha_{1-t}^2}$。这种过程在引入噪声的同时保持整体方差不变，这样能使数据的分布比较稳定。（可以看出 DDPM 就是这种过程）
- Optimal Transport Conditional: 定义均值和方差为 $\mu_t(x)=tx_1$、$\sigma_t(x)=1-(1-\sigma_\min)t$。可以求得最优传输路径是直线，因此可以更快地训练和采样。（这个比较类似于 Rectified Flow）

# Summary

Flow Matching 的确理论性较强，不是特别好理解。概括来说主要是给出了一种用来训练 CNF 的方法，并且提出了三个定理分别用来解决 flow 的表示问题、loss 函数的设计问题以及具体实现方式的问题。同时 flow matching 也统一了 score matching 和 DDPM，

> 参考资料：
>
> 1. [深入解析Flow Matching技术](https://zhuanlan.zhihu.com/p/685921518)
> 2. [【AI知识分享】你一定能听懂的扩散模型Flow Matching基本原理深度解析](https://www.bilibili.com/video/BV1Wv3xeNEds/)
> 3. [笔记｜扩散模型（一八）：Flow Matching 理论详解](https://littlenyima.github.io/posts/51-flow-matching-for-diffusion-models/)

