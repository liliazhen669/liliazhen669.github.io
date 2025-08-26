---
title: TransLight
author: liliazhen669
date: 2025-08-22 18:00:00 +0800
categories: [Learning, Relighting]
tags: [paper] # TAG names should always be lowercas
render_with_liquid: false
math: true
---

# Image-Guided Customized Lighting Control with Generative Decoupling

![fig-1](assets/img/translight/fig1.png)

## Abstract


现存的大多数光照编辑方法不能够提供定制的光影效果和保留内容整体性。这使得这些方法对于如何将推理图像的光照迁移到用户指定的图像这一复杂任务的表现很不好。为了解决这个问题，本文提出了 **TransLight**，一个能够实现高保真以及高自由度光影迁移的新颖方法。从推理图像中提取光影风格是本文最具挑战性的一步。难点在于光影中的复杂的几何结构特征是与真实世界图像的内容信息高度耦合的。为了实现这一目标，本文首先提出了生成式解耦技术，即使用两个经过微调的扩散模型来精确分离图像内容和光影，从而生成一个全新整理的、百万级的图像-内容-光三元组数据集。然后，本文采用 IC-Light 作为生成模型，并用三元组训练模型，并注入参考光照图像作为额外的调节信号。由此产生的 TransLight 模型能够实现各种光影的定制化和自然迁移。值得注意的是，通过彻底分离光影与参考图像，本文提出的生成式解耦策略赋予了 TransLight 高度灵活的照明控制能力。实验结果证明，TransLight 是首个能够成功在不同图像之间迁移光效的方法，它比现有技术能够提供更个性化的照明控制，并为照明协调和编辑领域的研究指明了新的方向。

## Introduction
