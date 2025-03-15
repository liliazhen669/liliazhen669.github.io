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

