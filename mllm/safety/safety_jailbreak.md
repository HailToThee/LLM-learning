# Safety & Jailbreak Attack（安全与越狱攻击）

> **目标**：绕过对齐/策略约束/内容护栏，诱导模型生成==禁止或有害==输出。
> 与完整性区别：完整性降级 correctness，越狱显式针对 ==harmlessness / policy compliance==。
>
> ==关键驱动==：安全对齐对**文本**最强，非文本模态（图/音/视）是防护薄弱通道，恶意意图可经此传递或混淆。

---

## 三条技术路线
1. **单模态面**：从单个防护薄弱的模态（如纯音频）发起
2. **多模态组合越狱**：跨模态组合，逃避单模态过滤器
3. **通用越狱触发器**：可复用、跨 prompt/任务通用的"万能钥匙"

---

## 基础方法：越狱优化（Jailbreak Optimization）

> 多模态越狱（如 Wang'24a、Geng'25）的优化框架直接继承自==文本越狱的 token 级优化==。这些方法给出"如何自动搜索越狱后缀/触发器"的范式。

### GCG: Greedy Coordinate Gradient (Zou et al., 2023)

Motivation:
1. 手工越狱模板（DAN 等）耗时且易被补丁覆盖。
2. 离散 token 空间无法直接用梯度下降。

Inspiration:
1. 用==梯度近似==评估每个 token 替换的收益，只在==最有希望的一个坐标==上做替换（贪心）。
2. 在有害 query 后接后缀 $S$，最大化模型对有害响应的似然：
$$\max_S\ \mathcal{L}_{\text{GCG}}(S) = \max_S\ \log P_{\text{target}}(\text{response}\mid \text{query}, S)$$
```
对每个候选 token: 用 ∇ 计算一阶近似收益 -> 选 top-k 候选 -> 逐个尝试 -> 保留最佳
```
3. 产出的后缀==可迁移==到其他模型，是多模态越狱优化的基础。

### AutoDAN (Liu et al., 2024)

Motivation:
1. GCG 的后缀是人类不可读的乱码，易被困惑度检测器识别。

Inspiration:
1. 用==分层遗传算法==搜索==语义可读==的越狱 prompt。
2. 交叉/变异保持句子流畅，同时最大化越狱成功率——绕过困惑度防御。

### PAIR: Prompt Automatic Iterative Refinement (Chao et al., 2024)

Motivation:
1. GCG/AutoDAN 需白盒梯度或大量计算。

Inspiration:
1. ==纯黑盒==——用一个攻击 LLM 迭代改写越狱 prompt，另一个目标 LLM 反馈是否成功。
2. 不需梯度，类似 ==LLM-as-attacker== 的对抗博弈。

### ICA: Iterative Contrastive Attack

Inspiration:
1. 对比学习思想：拉近有害响应、推远拒绝响应。
2. 迭代优化使模型偏好有害输出。

> ==与多模态的联系==：以上文本优化方法可==平移到视觉/音频 token 空间==（如 Geng'25 在嵌入空间对齐、Wang'24a 联合优化图文），即多模态越狱的技术底座。

---

## Type1: Unimodal Surfaces（单模态面）

单一模态发起，目标带共享语言解码核的 MLLM——==单模态编码器或其安全处理的弱点可危及整体系统==。

### Audio is the Achilles' Heel (Yang et al., 2024)

Motivation:
1. 音频模态安全防护远弱于文本。
2. ==文本安全与音频安全存在错配==。

Inspiration:
1. 用音频形式投放有害查询 + 语音专用越狱策略。
2. 证明文本-音频安全训练不一致。ASR ~70%。

### Multilingual and Multi-Accent Jailbreaking of Audio LLMs (Roh et al., 2025)

Motivation:
1. 跨语言音素/口音下的安全泛化能力未知。

Inspiration:
1. 多语言 + 多口音变体显著放大越狱成功率。
2. 说明安全训练与过滤在==跨语言音素/声学扰动==下泛化极差。ASR 提升 +57%。

---

## Type2: Multimodal Composite Jailbreaks（多模态组合越狱）

利用模态间交互（最常见图文/视频-文本），把恶意意图==分布或编码==，使单模态安全机制难以捕获。
> ==核心观察==：每个分量单独看都无害，融合后却有害。

### Visual Adversarial Examples Jailbreak Aligned LLMs (Qi et al., 2023)

Motivation:
1. 对齐过的 LLM 集成视觉后，视觉输入是否会绕过安全护栏？

Inspiration:
1. 视觉对抗样本可绕过对齐 LLM 的安全护栏。
2. ==单张对抗图可作为广泛有效的越狱工件==——一次优化，多查询通用。ASR 最高 91%。

### Images are Achilles' Heel of Alignment (Li et al., 2025b)

Motivation:
1. MLLM 有害性漏洞的主要来源尚不明确。

Inspiration:
1. 系统证据表明==图像是 MLLM 有害性漏洞的主要来源==。
2. 图像辅助越狱方法==放大==恶意意图。LLaVA ASR 90%，Gemini 72%。

### FigStep: Jailbreaking LVLMs via Typographic Visual Prompts (Gong et al., 2025)

Motivation:
1. 纯文本越狱已被安全微调覆盖。

Inspiration:
1. 把有害指令以==印刷文字形式渲染到图像==（视觉 prompt）。
2. LVLM 经 OCR 读入 $\to$ 绕过文本侧安全检查。平均 ASR 82.5%。

### Jailbreak VLMs via Bi-modal Adversarial Prompt (Ying et al., 2024)

Motivation:
1. 单模态扰动效果有限。

Inspiration:
1. ==联合优化==视觉与文本两路的双模态对抗 prompt。
2. 协调的图文操纵优于单模态扰动。

$$\min_{\delta_v,\delta_t}\ \mathcal{L}_{\text{jailbreak}}\big(F(I+\delta_v,\ T+\delta_t)\big)\quad\text{s.t.}\ \|\delta_v\|_\infty\le\epsilon_v,\ \|\delta_t\|_0\le k$$
MiniGPT-4 平均 ASR ~68%。

### VideoJail: Exploiting Video-Modality Vulnerabilities (Hu et al., 2025b)

Motivation:
1. 视频模态下，单图防御是否仍有效？

Inspiration:
1. 把恶意线索==分布到多帧== + 利用时序动力学。
2. 绕过对单图有效的防御。LLaVA-Video-7B 最高 96.5%。

### Visual Contextual Attack: Image-Driven Context Injection (Miao et al., 2025)

Motivation:
1. 纯黑盒、无梯度的越狱手段需求大。

Inspiration:
1. 以图像驱动构造==现实有害语境==的 context injection。
2. 黑盒 MLLM 上 ASR 85–91%。

### Jailbreak via Multi-Modal Linkage (Wang et al., 2025d)

Motivation:
1. 直接暴露有害内容易被检测。

Inspiration:
1. 跨模态==编码/解码==结构隐藏恶意意图。
2. 降低有害内容过度暴露，同时保持强越狱效果。ASR 最高 99%。

### Distraction is All You Need (Yang et al., 2025)

Motivation:
1. 模型对有害指令的检测注意力如何被分散？

Inspiration:
1. 把有害 prompt ==结构化分解== + 视觉增强分心。
2. 分散模型注意力，削弱检测/抑制能力。平均 ASR 52%，集成 74%。

### Medical MLLM is Vulnerable (Huang et al., 2025b)

Inspiration:
1. 医疗 MLLM 可被==跨模态攻击 + OOD 构造==越狱。
2. 白盒 82%，黑盒迁移 98.5%。

---

## Type3: Universal Jailbreak Triggers（通用越狱触发器）

产出==可复用==工件，跨 prompt/任务/输入泛化，作为 query-agnostic 的"万能钥匙"。
> 威胁更强：一次优化，反复部署。

### White-box Multimodal Jailbreaks Against LVLMs (Wang et al., 2024a)

Motivation:
1. 实例级越狱需==逐条优化==。

Inspiration:
1. 白盒下==联合优化图文组件==，产出通用 master key。
2. 对多种有害查询稳定触发有害肯定响应。

$$\min_{p^*,t^*}\ \sum_i \mathcal{L}_{\text{CE}}\big(F(p^*,t^*,q_i),\ \text{"affirm"}\big)$$
MiniGPT-4 ASR 96%。

### Con-Instruction: Universal Jailbreaking via Non-Textual Modalities (Geng et al., 2025)

Motivation:
1. 非文本模态是否能==自身编码==通用恶意指令？

Inspiration:
1. 优化对抗图像/音频，使其在嵌入空间对齐目标指令。
2. ==无需文本有害指令==即可绕过安全机制。灰盒 ASR 86.6%。

### Playing the Fool: OOD Strategy Jailbreak (Jeong et al., 2025)

Motivation:
1. 通用越狱也可源于==分布漂移==而非显式触发器优化。

Inspiration:
1. 对有害输入施加 OOD 变换，增加模型对恶意意图的==不确定性==。
2. 同时击败 LLM 与 MLLM 的安全对齐。LLaVA-1.5 13B 达 100%。

---

## 补充：prompt-to-image infection（已有笔记）

```
不需要梯度，而是用恶意文本调用扩散模型(如 Stable Diffusion)生成图像，再喂给 MLLM。
```
