# Integrity attack
### Type1: Signal perturbations

### Type2: Discrete Triggers
SCENETAP: Scene-Coherent Typographic Adversarial Planner against  Vision-Language Models in Real-World Environments

**previous backward:**
```
1. 当前的方法依赖于手动预定义的对抗性文本，无法适应不同的图像和问题，可能会降低攻击成功率。
2. 对抗性文本的放置遵循严格、预定义的模式（如居中或置于页边）
3. 这些攻击往往由于采用过于简单的放置策略且缺乏与场景的有机融合，从而导致视觉效果显得不自然
```
全流程ai，先理解生成文本精简，后分割图像找到合适编号，最后调用生成模型生成对应对抗图片。

**not good**

### Type3: Representation and Fusion exploits
表示与融合方法针对的是将模态特定编码器与语言解码器相连接的中间对齐机制。这些攻击并不单纯依赖于像素级的相似性约束，而是通过操纵联合嵌入、注意力模式或跨模态特征交互，使得即使看似无害的文本提示也会在对抗性的多模态语境下被错误解读。

Chain of Attack: On the Robustness of Vision-Language Models Against  Transfer-Based Adversarial Attacks

针对黑盒模型进行攻击，使用两个参考模型：
1. $M_{I2T}, M_{T2I}$ 给定一个图片，先根据$M_{I2T}$生成文本提示,根据文本提示来生成对抗文本，再根据$M_{T2I}$生成对应的图像表示。
2. Fusion representation $F = \alpha E_{v}(I) + (1 - \alpha) E_{t}(T)$, 由此生成干净样本的融合表示:$F$, 目标的表示:$F'$， 逐步通过添加扰动$\delta$来迭代。
$$
\begin{aligned}
\mathcal{L} &= \max\big(\text{sim}(F', F(x+\delta)) - \text{sim}(F(x+\delta), F) + \gamma, 0\big) \\
&= \text{ReLU}\big(\text{sim}(F', F(x+\delta)) - \text{sim}(F(x+\delta), F) + \gamma\big)
\end{aligned}
$$

再通过PGD来迭代更新$\delta$:
$$\delta_{t+1} = \text{Proj}_{\epsilon}\big(\delta_t + \alpha \cdot \text{sign}(\nabla_{\delta} \mathcal{L})\big)$$
其中$\text{Proj}_{\epsilon}$表示将扰动$\delta$限制在$\epsilon$范围内，$\alpha$是步长，$\gamma$是一个超参数，用于控制攻击的强度。

实际上都是类似于PGD的攻击方法，区别在于损失函数的设计，针对多模态模型的特征表示进行攻击，而不是直接对输入进行像素级的攻击。
 
Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models

拆分攻击步骤：
1. 无害的通用文本指令。
2. 对抗图像。

由于多模态模型的输入是图像和文本的组合，尤其是BEITv3直接讲图片和文本拼接在一起输入到Transformer中，因此攻击者可以通过设计特定的图像来干扰模型对文本的理解，从而实现攻击目的。

Algorithm:

$x_{adv}$ is the input of image we want to construct.
$x_{harmful}$ is the harmful image to trigger the attack.

1. $x_{harmful}-->H_{harmful}$, $x_{adv}-->H_{adv}$
2. L2 loss $L = L_{2}(H_{harmful}, H_{adv})$
3. $g = \nabla_{x_{adv}} L$
4. $x_{adv} = x_{adv} - \alpha \cdot g$

