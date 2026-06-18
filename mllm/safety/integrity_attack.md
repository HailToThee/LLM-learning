# Integrity attack（完整性攻击）

> **目标**：让模型输出**错误但看似合理**的内容（错误描述、幻觉、定向输出），且==不触发安全策略==。
> 与越狱的区别：越狱针对 harmlessness；完整性针对 ==correctness / perceptual grounding==，在良性 prompt 下操作。
>
> MLLM 管线：编码 $Z_i=E_i(X_i)$ → 融合 $Z_{\text{joint}}=f_{\text{fuse}}(Z_1,\dots,Z_n)$ → 解码 $Y=g(Z_{\text{joint}})$。
> 完整性攻击的==三层切入点==：(i) 连续信号扰动；(ii) 离散触发器；(iii) 跨模态表示/融合操纵。

---

## 基础方法 A：梯度类对抗攻击（Gradient-based）

> MLLM 的 Signal Perturbations（如 Dong'23、Chain of Attack）==本质都是这一族的变体==。
> 核心思想：实时计算模型对当前输入的梯度，沿"最能误导模型"的方向逐步加微小扰动。
> 关键特征：依赖梯度（白盒/可查询代理）、迭代优化、目标最大化损失、提升迁移性聚焦==改进梯度质量==。

### FGSM (Goodfellow et al., 2015)

Motivation:
1. 深度网络对输入虽高度非线性，但==局部近似线性==——梯度变化越线性，扰动可行性越高。
2. 基于泰勒一维展开，能否一步生成对抗样本？

Inspiration:
1. 单步沿梯度符号方向加扰动：
$$\delta = \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f_\theta(x), y))$$
$$x_{\text{adv}} = x + \delta$$
2. 快速但==迁移性弱==。

### I-FGSM / PGD (Madry et al., 2017)

Motivation:
1. FGSM 单步攻击强度不足，无法刻画最坏情况。

Inspiration:
1. 多步迭代 + 投影回 $\epsilon$-ball：
$$\delta_{t+1} = \Pi_\epsilon\big(\delta_t + \alpha \cdot \text{sign}(\nabla_\delta \mathcal{L}(f_\theta(x+\delta), y))\big)$$
2. $\Pi_\epsilon$ 投影算子，$\alpha$ 步长，通常迭代 10~40 步。提升攻击强度，刻画最坏情况上界。

### MI-FGSM (Dong et al., 2018)

Motivation:
1. PGD 易==陷入局部最优==，迁移性差。

Inspiration:
1. 引入==动量==累积历史梯度，==时间维度平滑==优化路径：
$$g_{t+1} = \mu \cdot g_t + \frac{\nabla_\delta \mathcal{L}}{\|\nabla_\delta \mathcal{L}\|_1},\quad \delta_{t+1} = \delta_t + \alpha \cdot \text{sign}(g_{t+1})$$
2. 跳出局部极值，提升迁移性。

### DI-FGSM (Xie et al., 2019)

Motivation:
1. 固定输入下梯度对特定像素过敏感。

Inspiration:
1. ==输入多样性==（随机缩放/填充），模拟不同输入变换，使梯度更鲁棒。==空间/输入维度梯度多样化==。

### TI-FGSM (Dong et al., 2019)

Motivation:
1. 梯度对特定像素位置过拟合，迁移到未对齐模型时失效。

Inspiration:
1. ==梯度卷积平滑==（Translation-Invariant），减少对特定像素位置的过拟合。
2. 可==预计算梯度==，无需目标模型梯度即可迁移，利于黑盒。

### Admix (Lin et al., 2020)

Inspiration:
1. ==混合多张图像==的梯度，增强泛化能力。

### 速查表

| 方法 | 核心技巧 | 目的 |
|------|----------|------|
| FGSM | 单步、符号梯度 | 快速，迁移性弱 |
| I-FGSM/PGD | 多步迭代+投影 | 提升攻击强度 |
| MI-FGSM | 动量累积 | 跳出局部极值，提升迁移性 |
| DI-FGSM | 输入多样性 | 梯度更鲁棒 |
| TI-FGSM | 梯度卷积平滑 | 减少像素过拟合 |
| Admix | 混合多图梯度 | 增强泛化 |

---

## 基础方法 B：对抗补丁与通用扰动（Patch & UAP）

> Discrete Triggers（如 Pandora's Box、SceneTAP）建立在这一族之上。
> 与梯度扰动不同：==强调可复用性==——一次构造，跨输入通用；对抗补丁还可物理打印、局部叠加。

### Adversarial Patch (Brown et al., 2017)

Motivation:
1. $L_p$ 约束的全局扰动在物理世界==难以实现==（打印/光照会破坏微小噪声）。
2. 需要一个可随处粘贴、物理鲁棒的攻击。

Inspiration:
1. 学习一个==局部 patch $p$==，可叠加到图像任意位置：
$$\max_{p}\ \mathbb{E}_{(x,\text{loc})}\big[\mathcal{L}(f(x \oplus_{\text{loc}} p), y_{\text{target}})\big]$$
2. $p$ 与输入无关、与位置无关，可物理打印。

### Universal Adversarial Perturbation / UAP (Moosavi-Dezfooli et al., 2017)

Motivation:
1. 逐样本优化代价高；能否==一个扰动欺骗大多数样本==？

Inspiration:
1. 求单一扰动 $\delta$ 使大部分数据被误分类：
$$\min_\delta \|\delta\|_2\quad\text{s.t.}\ f(x+\delta) \ne f(x)\ \text{for most } x \in \mathcal{D}$$
2. 揭示模型存在==通用表示漏洞==——正是综述 §4.2.5 指出的根因。

---

## 基础方法 C：生成式攻击（Generative model-based）

> 用==生成器 $G$== 直接输出扰动，训练时用代理模型提供梯度，==推理时无需访问目标模型==，天然黑盒。

### 通用框架

```
核心组件:
  G: 生成器(U-Net/ResNet encoder-decoder)，输入干净图 x，输出扰动 δ=G(x)
  f: 代理模型(白盒/多个)，提供梯度信号训练 G，推理时不需要
  L: 损失函数

基本流程:
  1. 特征提取:  频域分解/多尺度/文本条件注入
  2. 生成扰动:  δ=G(x), 施加范数约束 ||δ||∞≤ε
  3. 构造样本:  x_adv = x + δ
  4. 损失:      主攻击损失(最大化真实类分类损失) + 辅助对比/正则损失
  5. 优化:      仅更新 G，f 冻结; 训练后 G 直接黑盒攻击
```

### CDA: Cross-Domain Attack

Motivation:
1. 跨域迁移（如分类 $\to$ 检测）时，如何最大化 gap？

Inspiration:
1. 降低对抗样本对真实类的置信度——最大化交叉熵损失：
$$\max_\theta \mathcal{L}_{\text{CE}}(f(x_{\text{adv}}), y)$$
2. 用 $L_p$ 范数限制扰动范围。
3. **Relative Loss**（CDA 特有）：$\log \frac{f(x_{\text{adv}})_y}{f(x)_y}$，相对降低真实类置信度。

### FACL-Attack / CLIP-Guided / HGN

Inspiration:
1. **FACL**：提取中频/低高频分量——==拉近==非关键特征（低/高频）、==推远==关键语义特征（中频）。
2. **CLIP-Guided**：提取目标类别文本语义嵌入，跨模态对齐。
3. **HGN**：分别送入浅层/深层代理模型。

> 损失设计的一般规律：
> - **Attract（拉近）**：对抗样本与干净样本在==非关键特征==上相似
> - **Repel（推远）**：在==关键语义特征==上差异巨大

---

## Type1: Signal Perturbations（信号扰动）

对原始输入加==连续扰动==，诱导错误感知或下游推理错误。本质都是 PGD 系方法的变体，区别在**损失函数设计**——针对多模态特征表示，而非直接像素。

### How Robust is Google's Bard to Adversarial Image Attacks? (Dong et al., 2023)

Motivation:
1. 商业 MLLM（Bard/Bing/GPT-4V）是==黑盒闭源==，传统白盒对抗攻击无法直接应用。
2. 但它们复用了开源视觉编码器（如 CLIP），编码器的对抗脆弱性可能被继承到端到端系统中。

Inspiration:
1. 能否在==白盒代理视觉编码器==上优化扰动，再迁移到黑盒 MLLM？
2. 组件级扰动 $\to$ 完整助手泛化——证明编码器弱点会传播。

定向攻击：把图像嵌入拉向目标文本嵌入
$$\mathcal{L} = -\cos\big(E_v(I+\delta),\ E_t(T_{\text{target}})\big) + \lambda\cdot \text{TV}(\delta)$$
$$\delta_{t+1} = \Pi_\epsilon\big(\delta_t + \alpha\cdot\text{sign}(\nabla_\delta\mathcal{L})\big)$$
```
结果: Bard 22%, Bing 26%, GPT-4V 45%, ERNIE 86%
```

### On Evaluating Adversarial Robustness of LVLMs (Zhao et al., 2023)

Motivation:
1. 缺乏对开源 LVLM 的系统性对抗鲁棒性评估。
2. 攻击者往往拿不到 LLM 本身，只能碰视觉侧。

Inspiration:
1. targeted / untargeted 扰动即使==无 LLM 访问权限==也能操纵响应。
2. 扰动只需作用于视觉编码器侧——漏洞根源在编码器，而非语言解码。

### Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks (Qraitem et al., 2024)

Motivation:
1. LVLM 的 OCR/视觉-文本对齐模块会被印刷文字误导。
2. 但此前排版攻击依赖==人工设计==，无法自适应。

Inspiration:
1. What if 让 LVLM ==自己生成==欺骗性排版内容，反过来误导自己的感知？
2. 形成==级联完整性失败==：生成 → 误读 → 错答。

```
Pipeline: 模型生成排版文字 -> 叠加到图 -> OCR 重新读入 -> 感知被欺骗 -> 输出错误描述
```
准确率下降最高 60%。

### Attention! Your VLM Could be Maliciously Manipulated (Wang et al., 2025c)

Motivation:
1. 此前扰动多作用在"感知"层（让模型看错），很少能精确控制 token 级生成。

Inspiration:
1. 精心优化的图像扰动能否直接==操纵 token 级解码行为==？
2. 实现可控幻觉与细粒度输出操纵。越狱 ASR >88%，幻觉 >98%。

---

## Type2: Discrete Triggers（离散触发器）

依赖==结构化伪影==（patch、优化视觉图案、符号叠加）作为**可复用控制信号**。
> 与信号扰动的关键区别：强调==持久性与可复用性==——一次构造，跨输入/prompt/任务通用，无需逐样本优化。

### Pandora's Box: Universal Attackers Against Real-World LVLMs (Liu et al., 2024a)

Motivation:
1. 单实例对抗样本需==逐样本优化==，部署成本高。
2. 需要 task-agnostic 的通用攻击器，跨多个 LVLM 迁移。

Inspiration:
1. 构造与视觉编码器==稳定、输入无关==地交互的通用 patch $p$。
2. patch 让融合与生成系统性偏向攻击者选定解释。

$$\min_{p}\ \mathbb{E}_{(I,T)\sim\mathcal{D}}\big[-\text{sim}\big(f_{\text{fuse}}(E_v(I\oplus p),\ E_t(T_{\text{target}}))\big)\big]$$
对整个数据分布优化 $\Rightarrow$ 与输入无关（输入不可知）。语义相似度达 0.879。

### SceneTAP: Scene-Coherent Typographic Adversarial Planner (Cao et al., 2025)

Motivation:
1. 现有方法依赖==手动预定义==对抗文本，无法适应不同图像/问题，攻击成功率低。
2. 文本放置遵循==严格预定义模式==（居中/页边）。
3. 放置策略过简、缺乏与场景有机融合 $\to$ 视觉不自然，易被察觉。

Inspiration:
1. 全流程 AI 自动化——理解场景 → 生成精简文本 → 分割图像找合适位置 → 生成模型融合。
2. 满足==场景一致性 + 物理世界约束==，使对抗图在真实环境鲁棒。

```
Pipeline:
  1. VLM 理解场景 + 问题 -> 生成精简对抗文本
  2. 分割图像 -> 编号候选区域 -> 选最优放置位
  3. 调用生成模型 -> 融合生成场景一致对抗图
```
ASR 44%（MCQ）/62%（开放）。

---

## Type3: Representation and Fusion Exploits（表示与融合攻击）

针对==模态编码器 ↔ 语言解码器之间的中间对齐机制==。不依赖像素级相似约束，而是操纵联合嵌入 / 注意力 / 跨模态特征交互。
> 核心观察：使即使看似无害的 prompt，也会在对抗性多模态语境下被错误解读。

### Chain of Attack: Robustness of VLMs Against Transfer-Based Adversarial Attacks (Xie et al., 2024)

Motivation:
1. 黑盒目标模型无梯度可用，纯迁移攻击成功率有限。
2. 多模态语义关联可被利用来增强迁移。

Inspiration:
1. 用两个参考模型 $M_{I2T}, M_{T2I}$ 闭环——给定图像，$M_{I2T}$ 生成文本提示 $\to$ 生成对抗文本 $\to$ $M_{T2I}$ 生成对应图像表示，逐步迭代。
2. 攻击==融合特征表示==而非像素，更具迁移性。

先生成干净样本融合表示 $F$ 与目标表示 $F'$：
$$F = \alpha E_v(I) + (1-\alpha) E_t(T)$$
迭代优化 $\delta$，使扰动后表示==远离 $F$、靠近 $F'$==：
$$\mathcal{L} = \max\big(\text{sim}(F', F(x+\delta)) - \text{sim}(F(x+\delta), F) + \gamma,\ 0\big) = \text{ReLU}(\cdot)$$
PGD 迭代：
$$\delta_{t+1} = \text{Proj}_\epsilon\big(\delta_t + \alpha\cdot\text{sign}(\nabla_\delta\mathcal{L})\big)$$
```
本质: PGD 类方法，区别在损失针对融合特征表示而非像素。
定向 ASR 最高 98%。
```

### Jailbreak in Pieces: Compositional Adversarial Attacks on Multi-Modal LMs (Shayegani et al., 2023a)

Motivation:
1. BEITv3 等模型直接把图像和文本==拼接==输入 Transformer。
2. 攻击者可设计特定图像干扰模型对文本的理解。

Inspiration:
1. 把攻击==拆分==为两部分：(1) 无害通用文本指令；(2) 对抗图像。
2. 组合后触发攻击，每个分量单独看无害 $\to$ 逃避单模态检测。

```
Algorithm:
  x_adv: 待构造图像;  x_harmful: 触发攻击的有害图像
  1. 提取特征:  x_harmful -> H_harmful;  x_adv -> H_adv
  2. L2 损失:   L = L2(H_harmful, H_adv)
  3. 梯度:      g = ∇_{x_adv} L
  4. 更新:      x_adv = x_adv - α·g
```
ASR 85–87%（图像触发器）。

### VLAttack: Multimodal Adversarial Attacks via Pre-trained Models (Yin et al., 2023)

Inspiration:
1. 基于预训练模型同时对图像和文本两路生成扰动。
2. 利用==跨模态对齐通路==，从表示层注入。ASR 最高 93.5%。

### Break the Visual Perception: Attacking Encoded Visual Tokens (Wang et al., 2024b)

Inspiration:
1. 直接攻击 LVLM 编码后的==视觉 token==（而非原始像素）。
2. 从表示层注入，绕过像素级防御。ASR 最高 81.6%。
