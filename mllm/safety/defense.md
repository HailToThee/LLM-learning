# Defense Mechanisms（防御机制）

> 综述 §5 + 攻防综述。
>
> ==核心转变==：经典对抗鲁棒性聚焦"硬化模态编码器抵御有界扰动"；MLLM 部署引入新的==系统级失败模式==——不可信多模态内容（图中 OCR 文本、检索文档、工具输出）被误当作==高优先级指令==。
> 防御跨多层管线：感知层鲁棒性（完整性）→ 上下文/指令处理（越狱/注入）→ 控制面约束（工具 agent）→ 训练时投毒防护。

---

## Defense 1: Input Preprocessing & Perception-Layer Hardening（输入预处理 / 感知层硬化）

最直接抵御==完整性攻击（信号扰动）==。

### Feature Squeezing (Xu et al., 2018)

Motivation:
1. 小幅度扰动可利用视觉/音频输入处理弱点。

Inspiration:
1. 通过廉价输入压缩（位深降低、平滑）比较压缩前后预测，检测对抗样本。
2. 轻量、无需重训，但==对自适应攻击者保护有限==，且对不依赖像素/波形噪声的攻击无效。

### Adversarial Training / Robust Optimization (Madry'19; TRADES, Zhang'19)

Motivation:
1. 经验鲁棒性的基石——在最坏情况对抗样本上训练。

PGD 对抗训练（min-max）：
$$\min_\theta\ \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\max_{\|\delta\|\le\epsilon}\mathcal{L}(F(x+\delta;\theta), y)\big]$$

TRADES 形式化==精度-鲁棒权衡==：
$$\min_\theta\ \mathbb{E}_x\big[\underbrace{\mathcal{L}(F(x;\theta), y)}_{\text{自然精度}} + \beta\cdot\underbrace{D_{\text{KL}}(F(x;\theta)\|F(x+\delta;\theta))}_{\text{鲁棒性}}\big]$$

### Robust CLIP: Adversarial Fine-tuning of Vision Embeddings (Schlarmann et al., 2024)

Inspiration:
1. 对抗微调 CLIP 风格编码器。
2. 抵抗向下游 VLM 传播的视觉对抗扰动。

### Adversarial Prompt Tuning for VLMs (Zhang et al., 2024)

Motivation:
1. 改模型权重代价大。

Inspiration:
1. 调==prompt 表示==而非权重，无需改编码器即可提升鲁棒性。

$$\min_{P}\ \mathbb{E}_{(I,T),\delta}\big[\mathcal{L}(F(P,I+\delta,T), y)\big]$$

---

## Defense 2: Certified Robustness for Encoders（认证鲁棒性）

提供==形式化保证==：在指定扰动集内预测不变。

### Randomized Smoothing (Cohen et al., 2019)

Motivation:
1. 经验鲁棒无保证。

Inspiration:
1. 在 $\ell_2$ 扰动下提供==概率鲁棒证书==，可扩展到大模型。

对基分类器 $f$ 加高斯噪声 $\mathcal{N}(0,\sigma^2 I)$，预测类 $c$ 的认证半径：
$$R = \frac{\sigma}{2}\big(\Phi^{-1}(\underline{p_c}) - \Phi^{-1}(\overline{p_{\ne c}})\big)$$

### Fast Certification of VLMs (Nirala et al., 2024)

Inspiration:
1. 增量随机平滑认证多模态编码器在有界扰动下的鲁棒性。

### PromptSmooth: Certifying Medical VLM Robustness (Hussein et al., 2024)

Inspiration:
1. prompt 级认证策略用于医疗 VLM。

> ==局限==：认证最自然适用于单个感知组件。融合/自回归解码/指令遵循的==端到端认证仍开放==——表示级攻击与高层安全/控制攻击基本不在保证范围内。

---

## Defense 3: Multimodal Input Validation & Specification-Based Gating（输入验证 / 规约门控）

==MLLM 特有==：在允许输入影响推理前，按应用规约验证。

### Defending via User-Provided Specifications (Sharma et al., 2024)

Motivation:
1. 基于图像的 prompt 攻击针对 MLLM 聊天机器人。

Inspiration:
1. ==两阶段管线==：(i) 验证图像是否符合预期约束；(ii) prompt 注入防御，阻断图像编码的恶意意图。
2. 阻止不可信工件到达跨模态推理。

### Attack as Defense: Adversarial Perturbations Against Jailbreaking (Li et al., 2025a, EMNLP'25)

Motivation:
1. 硬过滤会损害良性体验。

Inspiration:
1. ==以攻为守==——主动引入对抗视觉扰动，干扰恶意指令，使其在影响 LLM 前失效。
2. 利用跨模态交互本身破坏越狱。

$$\max_{\|\eta\|\le\epsilon_d}\ \mathcal{L}_{\text{harm}}\big(F(I+\eta, T_{\text{jailbreak}})\big)$$

---

## Defense 4: Instruction–Data Separation（指令-数据分离）

> ==注入攻击的核心漏洞==：可信指令与不可信上下文（OCR 输出、检索文档、caption、工具结果）混合。
> 此层防御==结构化分离==，使不可信内容被当==数据==而非可执行指令。

### Formalizing & Benchmarking Prompt Injection (Liu et al., 2025b)

Motivation:
1. 朴素拼接使注入任务==覆盖==目标任务。

Inspiration:
1. 形式化 prompt 注入攻击与防御，提供基准评估框架。
2. 阐明为何简单拼接使注入任务覆盖目标任务。

### StruQ: Structured Queries (Chen et al., 2024)

Inspiration:
1. 显式==分离 prompt 通道与 data 通道==的结构化查询，提升对注入鲁棒性。

### Defensive Tokens / SecAlign (Chen et al., 2025a/b)

Inspiration:
1. **Defensive Tokens**：引入轻量防御 token，在混合可信/不可信上下文下提升抵抗力。
2. **SecAlign (CCS'25)**：用==偏好优化==塑模偏好，强化指令-数据分离。

偏好优化视角（对每个样本构造可信指令+不可信数据对）：
$$\min_\theta\ \mathbb{E}\big[-\log\sigma\big(\beta(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))\big)\big]$$
$y_w$ 为遵循可信指令的响应，$y_l$ 为被注入劫持的响应。

---

## Defense 5: Detection-Based Defenses（检测式防御）

识别不可信多模态上下文中嵌入的注入指令/恶意意图（含图中 OCR 文本）。

### PromptShield: Deployable Detection (Jacob et al., 2025)

Inspiration:
1. 为实时系统设计的==轻量分类器==，推理时检测 prompt 注入尝试。

### How NOT to Detect Prompt Injections with an LLM (Choudhary et al., 2025)

Motivation:
1. LLM-based 检测是否可靠？

Inspiration:
1. ==关键警告==——经验分析表明 LLM 检测在==自适应对手==下存在根本局限。
2. 攻击者可通过改写、间接化、多步指令分散==规避检测器==。
3. 检测须与分离/规约==互补==，不可单独依赖。

---

## Defense 6: Control-Plane Defenses for Tool-Using Agents（工具 Agent 控制面防御）

> MLLM 嵌入含工具的 agent 系统（浏览器/shell/检索/API）时，主导风险从"不安全文本"变为"==不安全动作=="。

### Cloak, Honey, Trap: Proactive Defenses Against LLM Agents (Ayzenshteyn et al., 2025, USENIX Security'25)

Motivation:
1. agent 的自主行为难以事先约束。

Inspiration:
1. 基于==欺骗与插桩==的主动防御——植入欺骗字符串、蜜标（honeytoken）、陷阱。
2. 检测并使自主 agent 行为脱轨。最直接对应控制面攻击（多模态输入作为对抗指令载体→工具误用）。

### AgentSentinel (Hu et al., 2025a)

Inspiration:
1. 端到端、实时安全防御框架，监控并干预计算机使用 agent。

### Agent Security Bench (ASB) (Zhang et al., 2025b)

Inspiration:
1. 基准驱动评估，形式化 LLM agent 中的攻击与防御动力学。

---

## Defense 7: Poisoning / Backdoor Defenses（投毒/后门防御）

数据投毒与后门驱动==数据整理期==与==训练后验证==双线防御。

| 方法 | 机制 | 阶段 |
|------|------|------|
| **Spectral Signatures** (Tran'18) | 在学习特征空间识别异常方向，过滤可疑训练点 | 数据整理 |
| **Neural Cleanse** (Wang'19) | 优化重建候选触发器→识别后门类→修复模型 | 训练后 |
| **STRIP** (Gao'19) | 强扰动下测预测一致性，运行时标记触发行为 | 推理时 |
| **Fine-Pruning** (Liu'18) | 剪枝干净输入下休眠的神经元+微调，削弱后门 | 训练后 |

### Neural Cleanse 数学公式

对每个可能的目标类 $t$，反向优化最小触发器 $m_t^*$：
$$m_t^* = \arg\min_{m}\ \mathcal{L}(F(x\oplus m;\theta), t) + \lambda\|m\|_1$$
检测触发器范数==异常大==的类即为后门类，随后通过神经元修剪/微调修复。

---

## 防御-攻击对应矩阵

| 防御层 | 主要抵御 | 局限 |
|--------|----------|------|
| 输入预处理/对抗训练 | 完整性（信号扰动） | 自适应攻击下部分有效 |
| 认证鲁棒性 | 编码器级完整性 | 端到端认证未解 |
| 输入验证/规约门控 | 越狱/注入（图像通道） | 规约窄时才适用 |
| 指令-数据分离 | 控制/注入 + 部分表示攻击 | 仅当攻击依赖上下文被解读为命令 |
| 检测式 | 注入/越狱 | 自适应攻击可绕过 |
| 控制面约束 | 工具/agent 注入 | 部署复杂 |
| 投毒/后门防御 | 训练时攻击 | 隐蔽后门（影子激活）难检测 |

> ==综述核心结论==：现有防御多针对特定攻击类或威胁模型，==跨模态/跨部署假设泛化有限==——未来工作的关键缺口。
