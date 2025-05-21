---
title: SpotLight
author: liliazhen669
date: 2025-05-20 12:00:00 +0800
categories: [Learning, Relighting]
tags: [paper] # TAG names should always be lowercas
render_with_liquid: false
math: true
---

# SpotLight: Shadow-Guided Object Relighting via Diffusion


## Abstract

扩散模型可以作为强大的神经渲染引擎以将虚拟对象插入图像中，但是和基于物理的渲染引擎相比，神经渲染引擎对光照设置的控制能力远远不足，而对于光照设置的控制通常对改善或个性化所需的图像结果至关重要。本文中，作者提出通过指定对象的阴影，进而实现对象重新照明这一思路。实验表明，仅将对象的阴影注入预先训练的基于扩散的神经渲染器，即可使其根据期望的光源位置准确着色对象，并在目标背景图像中正确和谐化对象（及其阴影）。此外，本文提出的方法实现了可控的重新照明结果，而无需额外训练。

## Introduction

该文介绍了一种名为SPOTLIGHT的方法，通过注入目标对象的阴影，使扩散模型能够实现精确的光照控制，实现对象重新照明。SPOTLIGHT在3D对象合成方面表现出色，优于其他支持光照控制的方法，使用了Amazon Berkeley Objects数据集和Laval Indoor HDR数据集。
 
**主要贡献**:
- SPOTLIGHT方法通过注入目标对象的阴影，使得扩散模型能够准确地根据期望的阴影位置对对象进行着色
- 通过两个并行分支接收不同方向的阴影，增强了对对象光照效果的控制。
- 在设计的实验中，SPOTLIGHT在3D对象合成方面表现出色，优于其他支持光照控制的方法。

## Method

利用用户提供的引导阴影来引导预训练扩散模型，从而实现插入物体的光照和谐。**核心想法**是：即使是近似的阴影也蕴含着重要的光照信息。通过在扩散模型的早期去噪阶段加入这一信息，可以同时优化合成外观及其阴影，使其与场景自然融合。本文提出的方法需要两个输入：物体掩码 $mask_{obj}$  和 由用于指定的引导阴影 $m_{shw}$。引导阴影通过fast rasterization计算得到或者直接由手绘得出，然后 $m_{shw}$ 会被用于引导隐扩散过程（latent diffusion process）。特别地，无需重新训练渲染器 $\mathcal{R}$，仅仅通过将引导阴影 $m_{shw}$ 融入到隐空间中便能得到和谐化后的结果。

### Background: diffusion renderers

（RGB -> X） 和 (Zerocomp) 证明了可以给定本征map（比如材质：albedo，roughness，matallic；几何：surface normals or depth 以及 shading），通过扩散模型来生成图像。作者将这类条件生成模型称为'扩散渲染器'。

去噪网络 $\mathcal{R}$ 在每一去噪时间步 $t$ 计算得到：

$$
\begin{equation}
\mathbf{v}_t=\mathcal{R}(\mathbf{\tilde{z}}_t,\mathbf{i},t),
\end{equation}
$$

其中 $\mathbf{v}_t$ 表示 $v-$预测， $\mathbf{\tilde{z}}_t$ 表示时刻 $t$ 的噪声编码， 以及 $\mathbf{i}$ 表示作为条件的本征map。

实验证明，这些方法能够以零样本的方式，将虚拟物体真实地合成到图像中，即使没有经过此任务的训练。只要提供虚拟物体的掩码及其本质map（由图形着色器渲染），就可以将该物体轻松地合成到背景图像的大多数本质map中（本质map是由经过训练的估计器预测的）。由于计算虚拟物体的阴影需要复杂的光照模拟，因此扩散方法通过提供部分shading map (Zerocomp) 来估计阴影，或训练修复模型（RGB -> X）。然而，与传统渲染不同，这些方法无法控制光照条件。

在本文中，作者使用了一个预训练的扩散渲染器，并展示了通过引入近似的引导阴影，来控制扩散模型以实现局部光照控制，类似于聚光灯围绕物体移动。下面是详述方法部分。

### Blending shadows

为了将引导阴影融入到隐表示中，需要将隐空间中每一时间步的噪声编码 $z_{t}$ 进行更新 (更新策略类似于(Blended latent diffusion 2023))：

$$
\begin{equation}
\mathbf{\tilde{z}}_t=(1-\beta\mathbf{m}_{\mathrm{shw},\downarrow})\odot\mathbf{z}_t+(\beta\mathbf{m}_{\mathrm{shw},\downarrow})\odot\mathrm{noise}(\mathcal{E}(\mathbf{g}),t)
\end{equation}
$$

更新后的 $\mathbf{\tilde{z}_t}$ 作为一种软约束，用于引导去噪过程以达到阴影一致的输出。在上述公式中，$\mathbf{g}$ 表示对象albedo，目标阴影和背景的合成，以及 $\beta$ 用于控制引导阴影的条件强度。$\mathbf{m}_{\mathrm{shw},\downarrow}$ 表示由目标掩码 $\mathbf{m}_{\mathrm{shw}}$ 下采样到隐空间得到隐空间中的阴影掩码。

### Shading the object via dual-branch guidance

本文方法的一个关键要素是通过CFG来放大引导阴影对object shading的影响。具体地，会处理两个扩散分支：融入desired shadow方向的正分支和压制undesired shadow方向的负分支（比如，通过一个相反的光照方向）。如图2所示，CFG机制会将这两个分支进行组合：

$$
\begin{equation}
\mathbf{\tilde{v}}_t=(1-\mathbf{m}_{\mathrm{obj},\downarrow})\odot\mathbf{v}_{t,\mathrm{pos}} +\mathbf{m}_{\mathrm{obj},\downarrow}\odot\left(\mathbf{v}_{t,\mathrm{neg}}+\gamma\left(\mathbf{v}_{t,\mathrm{pos}}-\mathbf{v}_{t,\mathrm{neg}}\right)\right),
\end{equation}
$$

其中 $\gamma$ 用来设置引导scale。这种双分支策略会强制物体上的预期照明，同时将其与背景自然融合

###  Final image synthesis

引导扩散过程结束后，使用 VAE 解码器对生成的潜变量进行解码。采用 (Zerocomp) 中的背景保留策略，确保仅修改目标及其阴影，从而生成具有自然局部光照的合成图像。
