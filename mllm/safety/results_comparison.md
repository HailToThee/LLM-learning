# 结果对比：攻击 ASR 与防御效果

> 数据来源：综述 A (arXiv:2603.27918) Table 6（65 篇实证工作）+ §5 防御讨论。
> ASR = Attack Success Rate（攻击成功率）。

---

## 一、攻击结果总表（按攻击家族分组）

### Integrity（完整性攻击）

| 论文 | 模态 | 攻击者知识 | 效果 |
|------|------|-----------|------|
| Dong'23 (Bard 攻击) | 图像 | 黑盒 | ASR: Bard 22%, Bing 26%, GPT-4V 45%, ERNIE 86% |
| Qraitem'24 (自生成排版) | 图+文 | 黑盒 | 准确率下降最高 60% |
| Liu'24a (Pandora's Box) | 图+文 | 黑盒 | 语义相似度达 0.879 |
| Cao'25 (SceneTAP) | 图+文 | 黑盒 | ASR 44%（MCQ）/62%（开放） |
| Xie'24 (Chain of Attack) | 图像 | 黑盒 | 定向 ASR 最高 98% |
| Shayegani'23a (Jailbreak in Pieces) | 图+文 | 黑盒 | ASR 85–87% |
| Yin'23 (VLAttack) | 图+文 | 黑盒 | ASR 最高 93.5% |
| Teng'25 (启发式风险分布) | 图+文 | 黑盒 | ASR 90%（开源）/68%（闭源） |
| Wang'25c (Attention 操纵) | 图像 | 白盒 | 越狱 ASR >88%, 幻觉 >98% |
| Wang'24b (视觉 token 攻击) | 图+文 | 灰盒 | ASR 最高 81.6% |
| Cui'24 | 图像 | 白盒 | Caption 检索 Recall 下降 90% |

### Safety & Jailbreak（安全与越狱）

| 论文 | 模态 | 攻击者知识 | 效果 |
|------|------|-----------|------|
| Qi'23 (视觉越狱) | 图+文 | 白盒 | 越狱 ASR 最高 91% |
| Wang'24a (白盒通用越狱) | 图+文 | 白盒 | ASR 96%（MiniGPT-4） |
| Ying'24 (双模态对抗 prompt) | 图+文 | 白盒 | MiniGPT-4 平均 ASR ~68% |
| Li'25b (图像是对齐软肋) | 图+文 | 黑盒 | ASR: LLaVA 90%, Gemini 72% |
| Gong'25 (FigStep 排版) | 图+文 | 黑盒 | 平均 ASR 82.5% |
| Miao'25 (图像驱动 context) | 图+文 | 黑盒 | ASR 85–91% |
| Wang'25d (多模态联动) | 图+文 | 黑盒 | ASR 最高 99% |
| Yang'25 (分心越狱) | 图+文 | 黑盒 | 平均 ASR 52%, 集成 74% |
| Jeong'25 (OOD 越狱) | 图+文 | 黑盒 | LLaVA-1.5 13B 达 100% |
| Cheng'25 (PBI-attack) | 图+文 | 黑盒 | ASR 最高 67.3% |
| Wang'25b (IDEATOR) | 图+文 | 黑盒 | ASR 最高 94% |
| Liu'24c (MM-SafetyBench) | 图+文 | 黑盒 | ASR 最高 72.14% |
| Liu'24d (Arondight) | 图+文 | 黑盒 | ASR 最高 84.50% |
| Lee'25 (视觉 prompt 注入) | 图+文 | 黑盒 | ASR 最高 90% |
| Yang'24 (音频越狱) | 音+文 | 黑盒 | 有害查询 ASR ~70% |
| Roh'25 (多语言多口音) | 音频 | 黑盒 | ASR 提升 +57% |
| Hu'25b (VideoJail) | 视频+文 | 黑盒 | LLaVA-Video-7B 最高 96.5% |
| Huang'25a (视频迁移) | 视频+文 | 黑盒 | MSVD-QA 55.48%, MSRVTT-QA 58.26% |
| Huang'25b (医疗 MLLM) | 图+文 | 白+黑 | 白盒 82%, 黑盒迁移 98.5% |
| Geng'25 (Con-Instruction) | 图+音 | 灰盒 | ASR 最高 86.6% |
| Hao'24 (多损失对抗搜索) | 图+文 | 白+黑 | ASR 最高 77.75%（MiniGPT-4） |
| Bailey'24 (Image Hijacks) | 图+文 | 白+黑 | ASR >80% |
| Wu'25 (Agent 鲁棒性解剖) | 图+文 | 白+黑 | ASR >67% |
| Zhang'25d (QAVA) | 图+文 | 白+黑 | InstructBLIP 从 78% 降至 44.85% |
| Aichberger'25 | 图+文 | 灰盒 | 通用攻击 100% |

### Control & Injection（控制与注入）

| 论文 | 模态 | 攻击者知识 | 效果 |
|------|------|-----------|------|
| Fu'23 (工具误用) | 图+文 | 白盒 | ASR 98% |
| Clusmann'25 (肿瘤学 VLM) | 图+文 | 黑盒 | GPT-4o ASR 67%（因模型而异） |
| Kimura'24 (目标劫持) | 图+文 | 黑盒 | ASR 最高 15.80% |
| Zhang'25c (弹窗攻击) | 图+文 | 黑盒 | 86% |
| Wang'25a (跨模态 agent 注入) | 图+文 | 黑盒 | 提升最高 30% |
| Zhang'25a (PoisonedEye RAG) | 图+文 | 黑盒 | 投毒成功率最高 92% |
| Liao'25 (EIA Web Agent) | 图+文 | 黑盒 | PII 泄漏达 70% |
| Hou'25 (音频注入) | 音+文 | 黑盒 | 防御成功率仅 3% |
| Tao'25 (ImgTrojan) | 图+文 | 黑盒 | AntiGPT prompt 83.5% |
| Gu'24 (Agent Smith) | 图+文 | 白盒 | 感染 ASR 接近 100% |

### Poisoning & Backdoor（投毒与后门）

| 论文 | 模态 | 攻击者知识 | 效果 |
|------|------|-----------|------|
| Xu'24 (Shadowcast) | 图+文 | 灰+黑 | 毒化 ASR >95%（label flip） |
| Lyu'24 (TrojVLM) | 图+文 | 白盒 | captioning ~0.97 |
| Lyu'25 (OOD 后门) | 图+文 | 黑盒 | 一致高 |
| Liang'24 (域漂移后门) | 图+文 | 黑盒 | 0.2% 投毒下 ASR >97% |
| Liang'25 (VL-Trojan) | 图+文 | 灰盒 | ASR 最高 99.82% |
| Yuan'25 (BadToken) | 图+文 | 白盒 | captioning ASR 100% |
| Liu&Zhang'25 (编码器后门) | 图像 | 灰盒/白盒 | ASR >99% |
| Liu'25a (LoRATK) | 文本 | 白盒 | ASR 95.8%–100% |
| Yin'25 (影子激活后门) | 图+文 | 白盒 | ASR 100% |
| Han'24 (四模态后门) | 图+文+音+视 | 白盒 | ASR >96% |
| Bagdasaryan'24 (对抗幻觉) | 图+文+音+热成像 | 白+黑+灰 | ImageBind/AudioCLIP ASR >99.5% |

---

## 二、跨维度统计（综述 A，65 篇）

### 按攻击目标
| 家族 | 篇数 | 占比 |
|------|------|------|
| Integrity | 20 | 31% |
| Safety & Jailbreak | 21 | 32% |
| Control & Injection | 14 | 22% |
| Poisoning & Backdoor | 11 | 17% |

> Tao'25 跨安全+投毒，统计中双计。

### 按攻击者知识
| 知识 | 篇数 | 解读 |
|------|------|------|
| 黑盒 | 36 | 最贴近真实部署 |
| 白盒 | 16 | 刻画最坏情况上界 |
| 灰盒 | 4 | 组件复用场景 |
| 混合 | 9 | 跨设置评估 |

### 按模态
| 模态 | 篇数 | 说明 |
|------|------|------|
| 图像/视频 | 58 | **绝对主导** |
| 纯非视觉 | 7 | — |
| 音频 | 9 | **研究不足** |
| 视频 | 4 | **研究不足** |

---

## 三、攻击"天花板"对比（最高 ASR）

| 攻击类型 | 代表方法 | 最高 ASR | 攻击者知识 |
|----------|----------|----------|-----------|
| 完整性 | VLAttack / Chain of Attack | 93–98% | 黑盒/灰盒 |
| 完整性(白盒) | Attention 操纵 | 幻觉 >98% | 白盒 |
| 越狱(黑盒) | 多模态联动 / IDEATOR | 94–99% | 黑盒 |
| 越狱(OOD) | Playing the Fool | 100% (LLaVA-1.5) | 黑盒 |
| 越狱(白盒通用) | White-box Multimodal | 96% (MiniGPT-4) | 白盒 |
| 控制(工具) | Misusing Tools | 98% | 白盒 |
| 控制(Agent) | Agent Smith | ~100% 感染 | 白盒 |
| 后门 | VL-Trojan / 影子激活 | 99.82–100% | 灰/白盒 |
| 跨模态(四模态) | 对抗幻觉 | >99.5% | 白盒 |

> **关键观察**：多数攻击在合适设置下 ASR 极高（>90%），说明**当前 MLLM 对齐在对抗场景下普遍脆弱**。

---

## 四、防御效果对比

| 防御方法 | 类型 | 主要抵御 | 已知效果/局限 |
|----------|------|----------|---------------|
| Feature Squeezing | 预处理 | 信号扰动 | 自适应攻击下部分有效 |
| PGD 对抗训练 | 训练 | 信号扰动 | 经典但代价高，精度-鲁棒权衡 |
| TRADES | 训练 | 信号扰动 | 显式建模精度-鲁棒权衡 |
| Robust CLIP | 编码器微调 | 视觉扰动向下游传播 | 提升下游 VLM 鲁棒性 |
| 对抗 prompt tuning | prompt | 视觉扰动 | 无需改权重 |
| Randomized Smoothing | 认证 | $\ell_2$ 扰动 | 概率保证，端到端未解 |
| PromptSmooth | 认证 | 医疗 VLM | prompt 级证书 |
| 规约门控 (Sharma'24) | 输入验证 | 图像越狱/注入 | 规约窄时适用 |
| Attack-as-Defense | 防御性扰动 | 越狱 | 以攻为守，干扰恶意指令 |
| StruQ | 指令-数据分离 | 注入 | 结构化查询 |
| SecAlign (CCS'25) | 偏好优化 | 注入 | 强化指令优先 |
| Defensive Tokens | token | 注入 | 轻量 |
| PromptShield | 检测 | 注入 | 实时，但自适应攻击可绕过 |
| Cloak/Honey/Trap | 控制面 | Agent | USENIX'25，主动欺骗 |
| AgentSentinel | 控制面 | 计算机 agent | 实时监控干预 |
| Spectral Signatures | 投毒检测 | 数据投毒 | 异常方向过滤 |
| Neural Cleanse | 后门检测 | 后门 | 触发器逆向+修复 |
| STRIP | 推理时 | 后门 | 预测一致性 |
| Fine-Pruning | 修复 | 后门 | 剪枝休眠神经元 |

---

## 五、攻防对照（哪个防御打哪个攻击）

| 攻击家族 | 主要可用防御 | 关键缺口 |
|----------|-------------|----------|
| Integrity（信号扰动） | 对抗训练、特征压缩、认证、Robust CLIP | 表示级攻击（非像素）覆盖不足 |
| Integrity（表示/融合） | 对抗 prompt tuning、Robust CLIP | 融合层攻击无专门防御 |
| Safety/Jailbreak | 规约门控、Attack-as-Defense | 通用/OOD 越狱（Jeong'25）几乎无防御 |
| Control/Injection | StruQ、SecAlign、检测、控制面 | Agent 注入（Agent Smith）防御薄弱 |
| Poisoning/Backdoor | Spectral、Neural Cleanse、Fine-Pruning | 影子激活后门（无触发器）难检测 |

---

## 六、研究空白（选题机会）

| 空白 | 证据 | 潜在方向 |
|------|------|----------|
| 🔥 视觉**推理过程**的安全 | 攻击多在感知层；推理链劫持（Stop Reasoning）刚起步 | 针对 CoT 推理的攻防 |
| 🔥 音频/视频模态 | 65 篇中音频仅 9、视频仅 4 | 视频/音频推理安全基准 |
| 🔥 "正确答案≠可靠推理" | 解耦假说在 MLLM 无系统研究 | 多模态推理鲁棒性评估 |
| 🔥 跨模态推理基准 | 评估协议碎片化 | 标准化多模态推理安全 benchmark |
| 防御泛化性 | 防御多针对特定攻击类 | 跨模态/跨场景通用防御 |
| 影子激活后门 | 无触发器，难检测 | 上下文驱动后门检测 |
| Agent 安全 | 单图越狱百万 agent | Agent 群体级防御 |

---

## 附：文件索引

- [survey_overview.md](survey_overview.md) — 综述总览与分类体系
- [integrity_attack.md](integrity_attack.md) — 完整性攻击
- [safety_jailbreak.md](safety_jailbreak.md) — 安全与越狱
- [control_injection.md](control_injection.md) — 控制与注入
- [poisoning_backdoor.md](poisoning_backdoor.md) — 投毒与后门
- [defense.md](defense.md) — 7 类防御
- [results_comparison.md](results_comparison.md) — 本文件
