# Control & Injection Attack（控制与注入攻击）

> **目标**：==覆盖指令优先级 / 执行逻辑 / 动作选择==，使系统遵循攻击者目标而非用户/系统指令。
> 与前两类区别：完整性降级 correctness，越狱绕过 safety，控制攻击==显式针对指令解释/优先级/执行机制==。
>
> ==最相关场景==：MLLM 嵌入含系统 prompt、工具调用、agent 控制环的交互系统——即使文本 prompt 良性，也可经多模态输入注入间接劫持。

---

## Type1: Prompt Injection（提示注入）

通过嵌入覆盖指令，与系统/开发者/用户指令竞争。
> 多模态下注入不必显式在文本中——可经==非文本模态==间接编码，影响模型对意图的内部解读。

### Abusing Images and Sounds for Indirect Instruction Injection (Bagdasaryan et al., 2023)

Motivation:
1. MLLM 的指令遵循特性把多模态上下文当作==权威来源==。
2. 文本侧过滤/审核可被绕过。

Inspiration:
1. 把恶意 prompt 对应的对抗扰动==混入视觉/音频输入==。
2. 诱导模型输出攻击者指定响应，即使没有任何显式指令文本。

### Prompt Injection Attacks on VLMs in Oncology (Clusmann et al., 2025)

Motivation:
1. 医疗 VLM（Nature Communications）部署中多模态注入的现实风险。

Inspiration:
1. 在图像/文档中隐藏注入指令。
2. GPT-4o ASR 67%（因模型而异）。

### Mind Mapping Prompt Injection (Lee et al., 2025)

Inspiration:
1. 现代 LLM 的视觉 prompt 注入攻击图谱化梳理。ASR 最高 90%。

---

## Type2: System Instruction Manipulation（系统指令操纵）

针对==高层控制信号==（工具选择逻辑、执行策略、agent 动作选择），而非用户指令。
> 危险在于：操作在==本应可信的层级==，直接影响模型与外部资源交互。

### Misusing Tools in LLMs with Visual Adversarial Examples (Fu et al., 2023)

Motivation:
1. 工具增强 LLM 中，视觉输入是否能触发==未授权的工具调用==？

Inspiration:
1. 视觉对抗样本可诱导攻击者期望的工具使用。
2. 操纵视觉输入即可让模型调用敏感工具（日历/信息检索），即使用户文本 prompt 无害。ASR 98%。

### Manipulating Multimodal Agents via Cross-Modal Prompt Injection (Wang et al., 2025a)

Inspiration:
1. 跨模态 prompt 注入操纵 agent 系统级决策逻辑。效果提升最高 30%。

---

## Type3: Tool, Retrieval, and Agentic Injection（工具/检索/Agent 注入）

扩展到 MLLM 作为==自主/半自主 agent== 运行：维护记忆、调用工具、检索外部上下文、多步推理。
> ==单次成功注入可跨多个下游动作传播==，影响被放大。

### Agent Smith: One Image Jailbreaks One Million MLLM Agents (Gu et al., 2024)

Motivation:
1. agent 部署中，单点注入的影响范围有多大？

Inspiration:
1. ==单张对抗图可同时越狱百万级多模态 agent==。
2. 利用 agent 间共享的感知与指令遵循机制，无需逐 agent 定制。

$$\max_{p^*}\ \frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\text{Jailbreak}(A_j(p^*))]\quad\text{s.t.}\ \|\cdot\|\le\epsilon$$
感染 ASR 接近 100%。

### PoisonedEye: Knowledge Poisoning on RAG-based LVLMs (Zhang et al., 2025a)

Motivation:
1. RAG-LVLM 的检索库可被污染。

Inspiration:
1. 投毒检索库，使 LVLM 在检索增强下产出攻击者控制的知识。投毒成功率最高 92%。

### EIA: Environmental Injection Attack on Web Agents (Liao et al., 2025)

Motivation:
1. 通用 web agent 面对真实网页环境的隐私泄露。

Inspiration:
1. 在网页环境注入对抗内容（弹窗/pop-up）。特定 PII 泄漏达 70%。

### Attacking VLM Computer Agents via Pop-ups (Zhang et al., 2025c)

Inspiration:
1. 用弹窗式视觉注入劫持视觉-语言计算机 agent。ASR 86%。

### Evaluating Robustness of Audio LMs to Audio Injection (Hou et al., 2025)

Inspiration:
1. 音频 LLM 的音频注入鲁棒性实证——==防御成功率仅 3%==（即攻击极易成功）。

---

## 与 tool_injection 的关系

> 原 `tool_injection.md` 为空。Tool/Agent 注入本质属于 Control 家族 Type3，故合并于此。
> 可保留 tool_injection.md 作 agent 专题索引，或删除。

## Control 攻击的核心漏洞链（综述 §4.4）

```
1. 指令遵循本性被利用 —— 模型"太听话"，无法区分意图权威性
2. 安全绕过          —— 多模态组合削弱单模态安全检查
3. 上下文操纵        —— 误导性背景信息
4. 注意力机制操纵    —— 劫持注意力权重
```
