# Poisoning & Backdoor Attack（投毒与后门攻击）

> **目标**：在==训练/指令微调/模型适配==阶段引入==持久性==漏洞。
> 与推理时攻击不同：植入的恶意行为由特定触发器或上下文激活，部署后长期存在。
>
> ==多模态更危险的原因==：(1) 复用预训练组件（视觉编码器）；(2) 依赖大规模网络数据；(3) 广泛使用指令微调/轻量 adapter 等高效适配方法。
> 三类切入点：数据投毒 / 后门 / 微调-adapter-表示投毒。

---

## Type1: Data Poisoning（数据投毒）

操纵训练/微调数据，在良性输入下诱导恶意行为，利用图文对齐植入微妙偏移，==不明显损害模型效用==。

### Shadowcast: Stealthy Data Poisoning Against VLMs (Xu et al., 2024)

Motivation:
1. VLM 训练数据来源复杂、规模大，==数据完整性假设脆弱==。
2. 此前投毒样本视觉上可辨别。

Inspiration:
1. 插入==视觉不可区分==的毒化图文对。
2. 少量毒化样本即可诱导持久恶意行为（误分类 + 说服行为），且==跨架构迁移==，在数据增强/压缩下仍有效。

双目标约束：(a) 触发时定向输出；(b) 视觉扰动不可见
$$\min_{\tilde{I},\tilde{T}}\ \mathcal{L}_{\text{task}}(\tilde{I},\tilde{T};y_{\text{target}}) + \lambda\cdot\|E_v(\tilde{I})-E_v(I_{\text{clean}})\|_2$$
毒化 ASR >95%（label flip）。

---

## Type2: Backdoor Attacks（后门攻击）

训练期植入隐藏触发器：干净输入下正常，触发器出现时产出攻击者指定输出。
> 多模态下触发器可嵌在==图像/指令/隐表示==中。

### TrojVLM: Backdoor Attack Against VLMs (Lyu et al., 2024)

Motivation:
1. 图像到文本生成任务的后门是否可行且语义合理？

Inspiration:
1. VLM 可被后门化，在图文生成中注入==预定义目标文本==，同时保持语义合理性。captioning ASR ~0.97。

### VL-Trojan: Multimodal Instruction Backdoor Against Autoregressive VLMs (Liang et al., 2025)

Motivation:
1. 指令微调的自回归 VLM 是否可被指令后门化？
2. 且在==有限攻击者访问==下？

Inspiration:
1. 指令微调阶段用==视觉与文本双触发器==植入多模态指令后门。
2. 即使攻击者权限有限也有效。ASR 最高 99.82%。

### Backdooring VLMs with Out-of-Distribution Data (Lyu et al., 2025)

Motivation:
1. 传统后门假设攻击者能访问原始训练分布——==现实中常不成立==。

Inspiration:
1. ==仅用 OOD 数据==即可植入后门，消除"访问训练分布"假设。
2. 即使投毒数据与部署输入分布不匹配，后门仍有效。

### ImgTrojan: Jailbreaking VLMs with One Image (Tao et al., 2025)

Motivation:
1. 跨家族——能否用==训练时后门==实现推理时越狱？

Inspiration:
1. ==单张图像==植入后门实现越狱。
2. 同时属于安全/越狱与投毒家族（综述中==唯一跨家族==）。AntiGPT prompt 下 83.5%。

### Shadow-Activated Backdoor Attacks (Yin et al., 2025)

Motivation:
1. 显式触发器易被检测。
2. 能否构造==无外部触发器==的后门？

Inspiration:
1. ==影子激活后门==——模型讨论特定对象/概念时自动激活恶意行为，无需任何外部触发器。
2. 激活由==上下文驱动==而非工件驱动，极大增加检测难度。白盒 ASR 100%。

---

## Type3: Fine-tuning, Adapter, and Representation Poisoning（微调/适配器/表示投毒）

针对==适配机制==（指令微调、token 级操纵、共享预训练编码器）。
> 利用即插即用微调与组件复用等广泛部署实践。

### BadToken: Token-Level Backdoor Attacks to MLLMs (Yuan et al., 2025)

Motivation:
1. 后门通常影响整体生成，==难以精细控制==。

Inspiration:
1. ==token 级后门==——遇到后门输入时插入/替换特定 token，操纵输出空间。
2. 保持整体效用的同时实现细粒度、隐蔽的生成控制。captioning ASR 100%。

### Stealthy Backdoor in Self-Supervised Vision Encoders (Liu & Zhang, 2025)

Motivation:
1. 视觉编码器常被多个 LVLM ==共享复用==。

Inspiration:
1. 直接在==自监督视觉编码器==中植入后门，后随编码器复用传播到众多下游 LVLM。
2. ==污染单一共享编码器==即可在下游模型诱发广泛幻觉与攻击者指定行为——无需修改语言模型本身。ASR >99%。

### LoRATK: LoRA Once, Backdoor Everywhere (Liu et al., 2025a)

Motivation:
1. LoRA adapter 在"分享即用"生态中广泛流通。

Inspiration:
1. LoRA adapter 本身可作为后门载体——==一次植入，随 adapter 分享到处生效==。ASR 95.8%–100%。

### Backdooring Multimodal Learning (Han et al., 2024)

Inspiration:
1. 跨图像/文本/音频/视频==四模态统一后门框架==。
2. 说明多模态联合学习会放大后门影响。ASR >96%。

---

## 跨家族协同

> ==重要观察（综述）==：攻击目标并非互斥。许多有效攻击是多目标的：
> - 完整性攻击常==赋能==越狱（损坏跨模态表示、削弱对齐）
> - 控制注入在输出层可能表现为完整性失败
> - 训练时后门在推理时表现为==定向完整性或控制违规==
>
> 综述对每个攻击只指定==首要目标==，但承认次要效果跨类。
> 例：ImgTrojan (Tao'25) 通过训练后门实现越狱，同时具备后门+越狱特征。
