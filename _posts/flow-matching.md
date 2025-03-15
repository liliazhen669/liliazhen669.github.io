---
title: Flow Matching 理论学习
author: liliazhen669
date: 2025-03-15 17:00:00 +0800
categories: [Learning, Diffusion Model]
tags: [flow matching] # TAG names should always be lowercas
render_with_liquid: false
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
其中 $t\in[0,1]$，$\mathbf{z}_t$ 称为**Flow Map**,或者**Transport Map**,可以理解为时刻 $t$ 下的数据点，$v(\mathbf{z}_t,t)$ 是一个向量场(Vector Field)，定义了每个数据点在状态空间中时刻 $t$ 下的变化大小与方向，通常由神经网络来学习。当神经网络完成了对向量场 $v(\mathbf{z}_t,t)$ 的学习后，就可以用如下的欧拉方法来求解：
$$
\mathbf{z}_{t+\Delta t}=\mathbf{z}_t+\Delta t\cdot v(\mathbf{z}_t,t)
$$
这意味着，给定一个初始概率分布（通常是标准高斯分布），可以通过学习向量场来学习数据的变换过程，从标准高斯分布采样，然后通过上述迭代过程得到目标分布中的一个近似解，完成生成的过程，从而实现从简单概率分布得到目标概率分布

## Continuity Equation

由于概率密度函数的性质确保了在全体分布上的积分为1，这反映了概率的总和是固定的，即概率是守恒的。在CNFs中，可以将这个性质与流体力学中的连续性方程类比，从而得到概率密度的连续性方程，这个方程可以写成：
$$
\frac{\partial p_t(\mathbf{x})}{\partial t}+\mathrm{div}(p_t(\mathbf{x})v_t(\mathbf{x}))=0
$$
其中 $p_t(\mathbf{x})$ 是时刻 $t$ 对应的概率密度函数、$v_t(\mathbf{x})$ 是与 $p_t(\mathbf{x})$ 关联的向量场。这个式子是向量场 $v_t(\mathbf{x})$ 能够产生概率密度路径 $p_t(\mathbf{x})$ 的充分必要条件，在后续的推导中会用这个式子作为一个约束来使用。

## Flow Matching

Flow Matching 的训练目标和 Score Matching 是比较类似的，学习的目标就是通过学习拟合一个向量场 $u_t$，使得能够得到对分布进行变换的概率路径 $p_t$，也就是下边这个公式：
$$
\mathcal{L}_\mathrm{FM}(\theta)=\mathbb{E}_{t,p_t(x)}||v_t(x)-u_t(x)||^2
$$
其中 $\theta$ 是模型的可训练参数，$t$ 在 0 到 1 之间均匀分布，$x\sim p_t(x)$ 是概率路径，$v_t(x)$ 是由模型表示的向量场。这个训练目标的含义为：利用模型 $\theta$ 来拟合一个向量场 $u_t(x)$，使得最终通过学习到的 $v_t(x)$ 可以得到概率路径 $p_t(x)$，并且满足 $p_1(x)\approx q(x)$。

