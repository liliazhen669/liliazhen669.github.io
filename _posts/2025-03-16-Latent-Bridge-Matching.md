---
title: Latent Bridge Matching for Fast Image-to-Image Translation
author: liliazhen669
date: 2025-03-16 16:00:00 +0800
categories: [Learning, Generative Models]
tags: [LBM] # TAG names should always be lowercas
render_with_liquid: false
math: true
---


> 论文链接：*[LBM: Latent Bridge Matching for Fast Image-to-Image Translation](https://arxiv.org/pdf/2503.07535)*


# Architexture

## Abstract

现有的扩散模型在img2img任务中需要多步才能达到比较好的效果，虽然已经有蒸馏或者流方法来加速采样过程，但是任然无法实现单步生成。因此，本文基于Latent Space 中的 Bridge Matching，来实现单步img2img

## Introduction and Related work

### Diffusion Models

- 迭代生成效率低：需要多步去噪（如50步），无法满足实时需求。
- 任务泛化性差：现有加速方法（如蒸馏）主要针对文本到图像任务，难以迁移到其他图像转换任务。

### Flow Matching and Bridge Matching：
- 像素空间计算成本高：直接在高分辨率像素空间建模导致计算复杂度高。
- 泛化能力受限：现有方法在低分辨率图像上表现良好，但难以扩展到高分辨率或复杂任务。

### 创新点与主要贡献创新点：

- 潜在桥匹配（LBM）：将桥匹配框架与潜在空间结合，解决高分辨率图像的计算瓶颈。

- 条件扩展：引入光照图等条件输入，支持可控图像生成（如阴影位置、光源颜色）。

## Method

### Bridge Matching

设 $\pi_{0}$ 和 $\pi_{1}$ 是两个概率分布，Bridge Matching的**主要思想**是，找到一个映射 $f$,使得能够从一个分布 $\pi_{0}$ 从采样得到样本 $x_{0}$ 后，通过映射 $f$ 得到另一个分布 $\pi_{1}$ 中的样本 $x_{1}$ 。因此为了达到这个目的，建立一个随机插值 $x_{t}$，使得在给定 $(x_{0}, x_{1})$ 的情况下, $x_{t}$ 的条件分布 $\pi(x_{t}|x_{0},x_{1})$ 本质上是一个布朗运动（也称为布朗桥），插值公式如下表示：

$$
x_{t} = (1-t)x_{0}+tx_{1}+\sigma\sqrt_{t(1-t)}\epsilon
$$

其中，$\epsilon \sim \mathcal{N}(0,I)$，$\sigma \ge 0$, 且 $t\in[0,1]$。值得注意的是，如果进一步设 $\sigma=0$，就可以得到流匹配公式，其可被视为Bridge Matching的零噪声极限。因此，$x_{t}$ 随时间的演化由以下随机微分方程（SDE）给出:

$$
dx_t = \frac{x_1-x_t}{1-t}d_t+\sigma dB_t,
$$

其中 $v(x_t,t)=(x_1-x_t) /(1-t)$ 被称为随机微分方程的**漂移项**。为了从分布 $\pi_{0}$ 中采样得到分布 $\pi_{1}$ 的样本，使用随机微分方程SDE时需要确保 $x_t(\pi_{t})$ 的分布是马尔可夫的，即不依赖于 $x_{1}$ 。在实际操作中，会进行马尔可夫投影，通常包括使用神经网络对随机微分方程的漂移项进行回归，训练目标为最小化如下函数：

$$
\mathbb{E}_{t,x_{0},x_{1}}\left [ ||(x_1-x_t) /(1-t)-v_{\theta}(x_t,t)|| \right ] 
$$

最后，估计出的漂移函数 $v_{\theta}$ 可以被整合到标准的随机微分方程求解器中，用于求解SDE，从而从分布 $\pi_{0}$ 中抽取的初始样本 $x_{0}$ 出发，生成服从分布 $\pi_{1}$ 的样本 $x_{1}$ 。
