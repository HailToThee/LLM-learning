# 多模态大模型安全：综述总览

> 整理自两篇核心综述：
> - **综述 A（攻击）**：*Adversarial Attacks on Multimodal Large Language Models: A Comprehensive Survey*, Jain et al., **TMLR 2026**, arXiv:2603.27918（37页, 88篇文献）
> - **综述 B（攻防）**：*A Survey of Recent Advances in Adversarial Attack and Defense on VLMs*, ScienceDirect 2026；以及武大+中科大+南洋理工 *LVLM 安全性综述*（攻击/防御/评估三分类，按模型生命周期）

---

## 一、两篇综述的核心定位

| 维度 | 综述 A（攻击） | 综述 B（攻防） |
|------|----------------|----------------|
| **立场** | 攻击为中心，解释"为什么能攻破" | 攻防对等，强调社会与安全意义 |
| **分类轴** | 攻击目标（goal-driven taxonomy） | 攻击面 + 防御层 + 评估 |
| **独特贡献** | vulnerability-centric 漏洞根因分析 | 按模型生命周期（pretrain→align→deploy）组织防御 |
| **数学化** | 把 MLLM 形式化为编码-融合-解码管线 | 偏方法学综述 |

**综述 A 的关键观点**：现有综述多按"模态/攻击面/应用"分类、只罗列技术；它改用**攻击目标**分类，并用**漏洞中心**视角把攻击映射到底层架构弱点——回答"为什么不同架构下同类攻击反复出现"。

---

## 二、MLLM 的形式化（综述 A, §2.2）

MLLM 处理 $n$ 个模态输入 $X=\{X_1,\dots,X_n\}$，产出 $Y$：

$$Y = F(X;\theta)$$

分三步：
1. **编码**：$Z_i = E_i(X_i;\theta_{E_i})$，模态特定编码器（ViT / 音频 Transformer）
2. **融合**：$Z_{\text{joint}} = f_{\text{fuse}}(Z_1,\dots,Z_n;\theta_{\text{fuse}})$
3. **解码**：$Y = g(Z_{\text{joint}};\theta_g)$

**对抗攻击目标**：找扰动 $\delta=\{\delta_1,\dots,\delta_n\}$，使 $X_{\text{adv}}=X+\delta$ 产生非预期输出，约束 $\|\delta_i\|_p \le \epsilon_i$（$p\in\{0,1,2,\infty\}$）。

三类目标函数：
- **非定向**：$\arg\max_\delta \mathcal{L}(F(X+\delta), Y_{\text{true}})$
- **定向**：$\arg\min_\delta \mathcal{L}(F(X+\delta), Y_{\text{target}})$
- **越狱**：$\arg\max_\delta P(C_{\text{harmful}} \in F(X+\delta))$

---

## 三、攻击分类体系（4 大家族）

```
MLLM 攻击
├── 1. Integrity（完整性）       — 输出错误/幻觉，不触发安全策略  → 20 篇
├── 2. Safety & Jailbreak（越狱）— 绕过对齐，生成有害内容        → 21 篇
├── 3. Control & Injection（注入）— 劫持执行逻辑/指令优先级      → 14 篇
└── 4. Poisoning & Backdoor（投毒）— 训练期植入持久后门           → 11 篇
```

> 论文分布（65 篇实证）：完整性 20 / 安全 21 / 控制 14 / 投毒 11（Tao et al. 跨安全+投毒）。

---

## 四、威胁模型交叉维度（攻击者知识）

| | 完整性 | 安全/越狱 | 控制/注入 | 训练时攻击 |
|---|---|---|---|---|
| **白盒** | Cui'24, QAVA'25 | Qi'23, Wang'24a | Fu'23, Bailey'24 | Lyu'24, Xu'24, Liang'24 |
| **灰盒** | Zhao'23 | Geng'25, Shayegani'23a | Wu'25 | Xu'24, Liu&Zhang'25 |
| **黑盒** | Zhao'23, Xie'24 | Gong'25, Jeong'25, Wang'25d | Clusmann'25, Kimura'24 | Xu'24, Liang'25 |

**统计（65 篇）**：
- 黑盒 36 篇（最贴近真实部署）、白盒 16、灰盒 4、混合 9
- **视觉攻击主导**：58/65 涉及图像/视频；音频 9、视频仅 4
- → 音频/视频安全是明确的研究空白

---

## 五、漏洞根因分析（vulnerability-centric，5 大类）

```
A. 跨模态交互与对齐
   A1 对齐/整合失败   A2 嵌入空间脆弱   A3 模态依赖不均衡
   A4 跨模态鲁棒性不对称   A5 通用表示漏洞
B. 模态特定处理
   视觉编码器继承脆弱性 / 音频防护弱 / OCR 误读 / 时序不一致
C. 指令遵循
   模型"太听话" / 安全绕过 / 上下文操纵 / 注意力操纵
D. 架构与组件
   预训练组件继承脆弱 / LoRA adapter 额外攻击面
E. 数据与训练
   投毒数据嵌入持久后门，跨模态学习放大影响
```

**关键洞察**：成功攻击往往**组合多个漏洞**，而非利用单一弱点。

---

## 六、配套文件索引

| 文件 | 内容 | 攻击家族 |
|------|------|----------|
| [integrity_attack.md](integrity_attack.md) | 信号扰动 / 离散触发器 / 表示-融合攻击 | Integrity |
| [safety_jailbreak.md](safety_jailbreak.md) | 单模态 / 多模态组合 / 通用越狱 | Safety |
| [control_injection.md](control_injection.md) | 提示注入 / 系统指令 / 工具-Agent 注入 | Control |
| [poisoning_backdoor.md](poisoning_backdoor.md) | 数据投毒 / 后门 / 微调投毒 | Poisoning |
| [defense.md](defense.md) | 7 类防御机制 | — |
| [results_comparison.md](results_comparison.md) | 攻击 ASR / 防御效果对比 | — |
