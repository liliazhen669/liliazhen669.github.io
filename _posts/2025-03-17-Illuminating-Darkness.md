---
title: Illuminating Darkness-Enhancing Real-world Low-light Scenes with
Smartphone Images
author: liliazhen669
date: 2025-03-17 16:00:00 +0800
categories: [Learning, Computational Photography]
tags: [Dataset] # TAG names should always be lowercas
render_with_liquid: false
math: true
---


> 论文链接：*[Illuminating Darkness](https://arxiv.org/pdf/2503.06898)*


# Flow Matching

## Abstract

数码相机难以在低光条件下照出好照片，为解决这个问题，本文提出了一个大规模高分辨率配对的Single-Shot Low-Light Enhancement（SSLLE）Dataset，通过Neural Network的方法来尝试解决这个问题。此外，本文还提出了通过一个tuning fork-shaped transformer model来分离地学习luminance和chrominance（LC）来增强低光图片，以解决复杂场景下的去噪以及过度增强问题。最后，本文还提出了一个用于特征融合的 LC 交叉注意力块，一个用于增强重建的 LC refinement 块，以及 能保证增强感知一致性的LC-guided 的监督学习方法 。 [项目地址](https://github.com/sharif-apu/LSD-TFFormer)

##
