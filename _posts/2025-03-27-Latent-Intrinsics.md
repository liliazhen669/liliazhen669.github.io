---
title: Latent Intrinsics Emerge from Training to Relight
author: liliazhen669
date: 2025-03-27 19:00:00 +0800
categories: [Learning, Computational Photography]
tags: [paper] # TAG names should always be lowercas
render_with_liquid: false
math: true
---


> 论文链接：*[Latent Intrinsics Emerge from Training to Relight](https://arxiv.org/pdf/2405.21074)*

> *[代码地址](https://github.com/xiao7199/Latent_Intrinsics?tab=readme-ov-file)* 

# Abstract

逆向图形方案可以恢复几何的显式表示和一组选择的内在特征，然后使用某种形式的渲染器进行照明。然而，逆向图形的误差控制很困难，而且逆向图形方法只能表示所选内在特征的影响。本文描述了一种完全基于数据驱动的照明方法，其中内在特征和照明分别表示为潜在变量。本文的方法产生了不错的实景照明效果，这是通过标准度量衡来衡量的。此外本文还展示了潜在内在特征可以恢复出反照率，而不需要使用任何反照率样例，并且恢复出的反照率与最先进的方法相当。

# Introduction

**解决问题**：本文旨在通过数据驱动的方法解决图像再照明问题，该问题是如何展示同一场景的图像在不同照明条件下的结果。与传统的逆图形方法不同，本文提出的方法不需要明确的几何表示和选定的内在属性，而是通过神经网络的方法将内在属性和照明表示为潜在变量，而不是传统重光照的那种光照相关变量的参数化建模。

# Method

本文的重光照模型可以视作一个自编码器：包含一个从给定目标场景计算潜在本质表示以及根据参考照明中的placeholder场景计算潜在外在表示的编码器，这两个表示被组合后会送入解码器解码成目前场景在参考照明下的图像。这一过程中应该施加如下两个限制：
- 最终呈现的目标场景在参考照明下的图像应该是正确的：场景中的物体应该不会变化
- 同一个场景在不同照明条件下的本质图像（albedo）也就是编码器计算得到的潜在本质表示，应该是不变的
设计这种**解耦结构**的目的是为了防止placeholder场景的本质特征泄漏到解码器中，进而影响目标场景在参考照明下最终的成像效果

