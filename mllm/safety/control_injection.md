# Control & Injection Attack（控制与注入攻击）

> **目标**：==覆盖指令优先级 / 执行逻辑 / 动作选择==，使系统遵循攻击者目标而非用户/系统指令。
> 与前两类区别：完整性降级 correctness，越狱绕过 safety，控制攻击==显式针对指令解释/优先级/执行机制==。
>
> ==最相关场景==：MLLM 嵌入含系统 prompt、工具调用、agent 控制环的交互系统——即使文本 prompt 良性，也可经多模态输入注入劫持。

---

## 基础方法：经典提示注入（Classical Prompt Injection）

> 多模态注入（Bagdasaryan'23、Clusmann'25）的内核直接继承自==文本 LLM 的提示注入==。这些方法定义了"如何让模型遵循攻击者指令而非用户/系统指令"。

### Ignore Previous Instructions (Perez & Ribeiro, 2022)

Motivation:
1. LLM 把所有输入当指令流，==无法区分指令来源的权威性==。

Inspiration:
1. 注入"忽略之前的指令"类覆盖语句，劫持模型行为：
$$\text{Prompt}: \underbrace{\text{system/user instruction}}_{\text{合法}}\ +\ \underbrace{\text{Ignore previous... do X}}_{\text{注入}}$$
2. 系统指令可被用户文本覆盖。提示注入一词的起源。

### Goal Hijacking (Perez & Ribeiro, 2022)

Motivation:
1. 直接覆盖系统指令有时不稳定；能否让模型"偏题"到攻击者目标？

Inspiration:
1. 把模型从==原任务==劫持到==攻击者目标任务==，而非单纯忽略。
2. 分类：full goal hijacking（完全劫持）、partial（部分）。更隐蔽的注入形式。

### DAN / Role-play Jailbreak

Motivation:
1. 显式有害请求会被安全对齐拒绝。

Inspiration:
1. ==角色扮演==（Do Anything Now / 虚构人格）绕过安全检查。
2. 让模型进入"无限制"人格，间接生成有害内容。越狱的早期模板。

### Indirect Prompt Injection

Motivation:
1. 直接注入在用户输入里，易被检测。

Inspiration:
1. 把注入藏在==模型检索/读取的外部内容==中（网页、文档、图像 OCR 文本）。
2. 模型把不可信外部内容当可信指令执行——==MLLM 多模态注入的直接前身==（图像/音频即"外部内容"载体）。

> ==与多模态的联系==：经典注入的"覆盖→劫持→角色扮演→间接注入"四步，对应 MLLM 中==非文本模态作为间接注入载体==（图/音频/工具输出隐藏指令），是 control 家族的方法底座。

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
