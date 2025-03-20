---
title: LumiNet-Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting
author: liliazhen669
date: 2025-03-18 18:00:00 +0800
categories: [Learning, Relighting]
tags: [paper] # TAG names should always be lowercas
render_with_liquid: false
math: true
---


> 论文链接：*[LumiNet](https://arxiv.org/pdf/2412.00177)*


# Flow Matching

## Abstract

本文提出了一个利用生成模型和潜在本质以实现高效光照传输的新颖模型。本文提出的方法有两个主要贡献点：基于StyleGan重光照模型的数据构造策略以及能够处理来自源图像的潜在本质表示和来自目标图像的潜在外在表示的ControlNet。本文还通过一个将目标图像的潜在外在表示通过交叉注意力和微调UNet的方式插入到去噪网络的可学习的MLP来增强光照传输。具体来说，LumiNet的目标是在保持源场景的几何结构（Geometry）和材质（Material）的同时，将目标光照图像的光照特性转移到源图像上，生成逼真的重光照效果

## Introduction And Related Work

### 解决的过去方法的问题

传统方法的局限性：

- 传统逆渲染方法：传统方法通常通过逆渲染（inverse rendering）来分解场景的光照、材质和几何信息，然后重新渲染。然而，这些方法通常依赖于显式的3D重建和材质建模，计算成本高且容易受到模型误差的影响。
- 多视图方法：一些方法依赖于多视图图像来获取更丰富的场景信息，但这需要复杂的设备设置和大量的数据采集工作。
- 单图像方法：单图像方法虽然简化了输入要求，但在处理复杂的光照现象（如镜面高光、间接光照等）时表现不佳，尤其是在不同场景之间进行光照转移时。
现有生成模型的局限性：
- StyleGAN：虽然StyleGAN的潜在空间可以解耦光照信息，但其生成的图像与真实图像之间存在差距，难以直接应用于真实图像的重光照任务。
- 扩散模型（Diffusion Models）：现有的扩散模型在生成条件图像方面表现出色，但在处理复杂室内场景的光照转移时，仍然存在局限性，尤其是在跨场景光照转移时。

### 创新点和主要贡献

- 新颖的框架：LumiNet结合了潜在内在控制和扩散模型，能够在不需要3D或多视图输入的情况下，实现高质量的室内场景重光照。
- 据生成策略：通过变分StyleGAN方法，LumiNet能够生成多样化的训练数据，解决了真实室内场景在不同光照条件下的数据稀缺问题。
- 泛化能力：尽管LumiNet仅在相同场景的光照对上进行训练，但它能够成功地在不同布局和材质的场景之间进行光照转移。
- 逼真的光照效果：LumiNet能够生成复杂的光照现象，如镜面高光、投射阴影和间接光照，生成的结果在物理上具有合理性。

## Method