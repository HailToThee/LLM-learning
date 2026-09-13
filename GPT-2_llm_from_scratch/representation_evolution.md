# 文本表示方法的演进:one-hot → TF-IDF → n-gram → word2vec → ELMo → GPT

> 主线:每一种方法都在**修补前一种的缺陷**。
> 离散稀疏表示 → 稠密分布式表示 → 上下文相关表示 → Transformer 自回归预训练。
>
> 本笔记配套 Raschka《Build a Large Language Model (From Scratch)》ch2~ch7。

| 方法 | 年份 | 表示类型 | 是否含语义 | 是否依赖上下文 | 核心缺陷 |
|---|---|---|---|---|---|
| one-hot | — | 离散、稀疏 | ✗ | ✗ | 维度爆炸、词间无相似度 |
| TF-IDF | 1970s | 稀疏加权 | ✗ | ✗ | 词袋、丢失语序 |
| n-gram LM | 1980s~90s | 统计概率 | ✗ | 局部 | 维度灾难、长程弱 |
| word2vec | 2013 | 稠密向量 | ✓ | ✗(静态) | 一词一向量、无法消歧 |
| ELMo | 2018 | 上下文向量 | ✓ | ✓ | LSTM、难并行 |
| GPT | 2018+ | Transformer 上下文 | ✓ | ✓ | (自此开启大模型时代) |

---

## 1. One-Hot(独热编码)—— 最朴素的离散表示

给定词表 $V$,每个词映射为一个长度 $|V|$ 的向量,只有该词对应位置为 1,其余为 0:

$$w_i \;\mapsto\; \mathbf{e}_i = (0,\dots,0,\underbrace{1}_{i},0,\dots,0) \in \{0,1\}^{|V|}$$

**性质**:任意两个不同词的内积 $\mathbf{e}_i^\top \mathbf{e}_j = 0$,即**所有词两两正交、毫无相似度可言**。

```
"king"   -> [1, 0, 0, ..., 0]
"queen"  -> [0, 1, 0, ..., 0]
"apple"  -> [0, 0, 1, ..., 0]
# d(king, queen) == d(king, apple)  全部相等
```

**缺陷**:
1. 维度 $O(|V|)$,词表大时存储/计算爆炸;
2. 没有任何语义信息,无法泛化(同义词、类比关系都丢失);
3. 完全稀疏,下游模型学不到有用梯度。

> 一句话:**有区分度,无语义。** 它只是"身份编号",不是"含义"。

---

## 2. TF-IDF —— 给词袋加权的统计表示

TF-IDF 仍是**词袋(Bag-of-Words)**,但用统计权重衡量一个词对一篇文档的重要性:

$$\text{tf-idf}(t, d) = \underbrace{\text{tf}(t,d)}_{\text{词频}} \times \underbrace{\log \frac{N}{\text{df}(t)}}_{\text{逆文档频率 idf}}$$

- $\text{tf}(t,d)$:词 $t$ 在文档 $d$ 中出现次数;
- $\text{df}(t)$:包含词 $t$ 的文档数;$N$ 为总文档数。
- 直觉:**常见词**(the、of)idf 小、权重低;**文档特有词**idf 大、权重大。

文档表示 = 把每个词的 tf-idf 拼成稀疏向量。

**相比 one-hot 的进步**:区分了"停用词"和"关键词"。
**仍有的缺陷**:
1. 仍是稀疏、高维($|V|$ 维);
2. **词袋模型**,丢弃语序("dog bites man" 与 "man bites dog" 表示相同);
3. 没有语义,无法捕获同义/相关。

> 一句话:**TF-IDF 解决了"哪些词重要",但没解决"词之间什么关系"。**

---

## 3. N-gram 语言模型 —— 用统计做"下一个词预测"

n-gram 是最早的**语言模型**:基于马尔可夫假设,用前 $n-1$ 个词预测第 $n$ 个词:

$$P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1}) = \frac{\text{count}(w_{t-n+1}, \dots, w_t)}{\text{count}(w_{t-n+1}, \dots, w_{t-1})}$$

整个句子的概率(链式法则 + 马尔可夫截断):

$$P(w_1, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$$

**进步**:第一次把"预测下一个词"作为目标 —— 这正是后来所有神经语言模型(GPT)的训练信号。

**致命缺陷**:
1. **数据稀疏**:未在语料中出现的 n-gram 概率为 0,需平滑(smoothing,如 Laplace / Kneser-Ney);
2. **维度灾难**:n 增大时,参数量 $O(|V|^n)$ 指数爆炸,实际只能用到 3~5-gram;
3. **长程依赖弱**:n-gram 只看前 $n-1$ 个词,无法建模远距离关系;
4. **无语义泛化**:"apple" 和 "orange" 被当作完全不同的符号,统计上互不共享。

> 一句话:**n-gram 给出了正确的目标(预测下一个词),但用计数做统计走到了尽头 —— 需要把离散符号换成连续向量,这就是神经语言模型 + word2vec 的起点。**

```
关键转折(Bengio 2003, NPLM):
  用一个共享的稠密词向量矩阵 lookup,再用神经网络算 P(next|context)。
  训练完后,这个 lookup 矩阵就成了"有语义的词向量" —— word2vec 正是把它单独抽出来、极致加速。
```

---

## 4. Word2vec —— 稠密分布式表示的开端(重点)

**核心思想**:把每个词表示成一个**低维稠密向量**(如 300 维),词义**分布式**地编码在所有维度上。向量从"预测上下文"的自监督任务中自动学出来。

两种对称架构:
- **CBOW**(Continuous Bag-of-Words):用上下文词预测中心词;
- **Skip-Gram**:用中心词预测上下文词。小数据集 / 低频词场景下 Skip-Gram 更好。

### 4.1 CBOW
用上下文词向量$o$平均或加权求和,预测中心词$c$:
$$P(w_c \mid w_{o_1}, \dots, w_{o_{2m}}) = \frac{\exp(\mathbf{v}'_{w_c}{}^\top \hat{\mathbf{v}})}{\sum_{w=1}^{W} \exp(\mathbf{v}'_{w}{}^\top \hat{\mathbf{v}})}$$

其中，平均上下文向量 $\hat{\mathbf{v}}$ 的公式为：
$$\hat{\mathbf{v}} = \frac{1}{2m} \sum_{i=1}^{2m} \mathbf{v}_{w_{o_i}}$$
### 4.1 Skip-Gram 原理

给定中心词 $c$ 和窗口内的上下文词 $o$,最大化:

$$P(o \mid c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w=1}^{W} \exp(\mathbf{u}_w^\top \mathbf{v}_c)}$$

其中 $\mathbf{v}_c$ 是中心词向量(输入矩阵),$\mathbf{u}_o$ 是上下文词向量(输出矩阵),两套向量(word2vec 最后通常取 $\mathbf{v}$ 或取平均)。

**痛点**:分母 $\sum_{w=1}^{W}$ 要遍历整个词表,复杂度 $O(W)$,训练 Wikipedia 量级语料慢到不现实。

### 4.2 负采样(Negative Sampling)—— 把多分类降维成二分类

不再问"哪个词是上下文"(多分类),改成问"这个(中心, 目标)配对是真上下文还是瞎凑的"(二分类):

$$\mathcal{L} = -\Big[\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c) + \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_c)\Big]$$

- 第一项:正样本(真实上下文对),希望内积大;
- 第二项:$K$ 个负样本(按 $P_n(w) \propto \text{count}(w)^{0.75}$ 采样),希望内积小。
- 复杂度从 $O(W)$ 降到 $O(K+1)$,$K\approx 5\sim 20$,数量级提速。

> 这也是为什么输入从"中心词"变成"(中心, 目标)**pair**":二分类的判断单位本来就是一对词的关系。

(另一个等价方案:**Hierarchical Softmax**,用霍夫曼树把 $O(W)$ 压成 $O(\log W)$。)

### 4.3 学到的性质

- **语义相似度**:用余弦相似度衡量,$\cos(\mathbf{v}_a, \mathbf{v}_b) = \frac{\mathbf{v}_a^\top \mathbf{v}_b}{\|\mathbf{v}_a\|\|\mathbf{v}_b\|}$;
- **线性类比关系**:著名的 $\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}$;
- 向量空间的**方向**编码语义关系(性别、时态、国家-首都)。

### 4.4 核心缺陷 —— 静态

word2vec 是**静态词向量**:每个词无论出现在哪,都是**同一个固定向量**。
- "bank" 在 "river bank" 和 "bank account" 里向量相同,无法消歧;
- 无法处理多义词、一词多类。

> 一句话:**word2vec 把"符号"变成了"有语义的稠密向量",但它是静态的、上下文无关的 —— 这个缺陷由 ELMo / BERT 解决。**

---

## 5. ELMo —— 上下文相关的深度表示(重点)

**全称**:Embeddings from Language Models(Peters et al., 2018, AllenAI)。
**要解决的问题**:word2vec 的静态性。

ELMo 的核心:**每个词的向量是整个句子的函数**,同一个词在不同句子里向量不同。

### 5.1 结构:双向多层 LSTM 语言模型(biLM)

```
                  ┌─ Forward  LSTM (layer 2) ─┐
 字符级 CNN ──→    ├─ Forward  LSTM (layer 1) ─┤  → 正向隐状态 h^f
 (char-CNN)   ──→  ┤                           │
                  ├─ Backward LSTM (layer 1) ─┤  → 反向隐状态 h^b
                  └─ Backward LSTM (layer 2) ─┘
```

1. **输入层:字符级 CNN** —— 把词拆成字符卷积。好处:能处理**未登录词(OOV)**和词缀信息(un-、-ing);
2. **正向 LSTM**:从左到右,根据前文预测下一个词;
3. **反向 LSTM**:从右到左,根据后文预测上一个词。

### 5.2 向量怎么算 —— 多层表示加权融合

对句子中第 $k$ 个词,模型输出 $2L+1$ 个表示($L$ 层 LSTM × 正反两个方向 + 输入层)。ELMo 向量是它们的**线性加权**:

$$\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \, \mathbf{h}_{k,j}$$

- $s_j$:每层权重,**对下游任务单独学习**(可训练);
- $\gamma$:全局缩放因子。
- 直觉:**浅层偏词法/句法,深层偏语义**,下游任务自己决定各层权重。

### 5.3 训练目标

双向语言模型损失:正向预测下一个词 + 反向预测上一个词(都是 next-token 式的交叉熵),在海量文本上**预训练一次后冻结**。

### 5.4 用法:Feature-based(特征拼接)

| | ELMo | BERT |
|---|---|---|
| 骨干 | biLSTM | Transformer |
| 使用方式 | **冻结 + 拼接**(feature-based) | **整体微调**(fine-tune-based) |
| 上下文 | ✓ | ✓ |

ELMo 当年大幅刷新了 NER、情感分析、SQuAD 等 SOTA,终结了静态词向量时代。

### 5.5 核心缺陷 —— LSTM 的天花板

1. **难以高效并行**:LSTM 逐步递推,GPU 并行度低;
2. **长距离依赖弱**:梯度链长,远距离信息衰减;
3. 特征式用法,不如微调式端到端。

> 一句话:**ELmo 把词向量从"静态"升级成"上下文相关",证明了预训练表示可大规模迁移;但 LSTM 是它的天花板 —— 换成 Transformer 就是 GPT/BERT。**

---

## 6. GPT —— Transformer + 自回归预训练(重点)

**全称**:Generative Pre-trained Transformer(OpenAI, 2018~)。
**两个关键转变**:
1. **骨干**:biLSTM → **Transformer**(self-attention),可全并行 + 长程依赖强;
2. **范式**:大规模**自监督预训练 + 下游微调/提示**(Pretrain then Transfer)。

### 6.1 训练目标:预测下一个词(Causal LM)

给定序列 $w_1, \dots, w_T$,最大化:

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_{<t};\theta)$$

> 注意:**GPT 是单向的(decoder-only)**,位置 $t$ 只能看到 $<t$ 的词 —— 这正是自回归(autoregressive)。这也是它和 BERT(双向 MLM)的根本区别。

### 6.2 为什么用 Transformer 而不是 LSTM

| | LSTM | Transformer(self-attention) |
|---|---|---|
| 并行性 | 必须逐步递推,无法并行 | 所有位置**同时**计算,全并行 |
| 长程依赖 | 链式梯度,远距离衰减 | 任意两词**一步直连**,路径长度 $O(1)$ |
| 归纳偏置 | 强序列偏置(利于小数据) | 异偏置,**靠规模和数据驱动** |

弱归纳偏置 + 大数据/大参数 → 表现随规模持续上升,这正是 GPT-2/3 "Scaling Law" 的基础。

### 6.3 Self-Attention(GPT 的核心计算)

给定输入序列表示 $\mathbf{X} \in \mathbb{R}^{T\times d}$,线性投影出 Q/K/V:

$$\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,\quad \mathbf{K}=\mathbf{X}\mathbf{W}_K,\quad \mathbf{V}=\mathbf{X}\mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

- $\mathbf{Q}\mathbf{K}^\top$:每个位置对所有位置算**相似度(注意力权重)**;
- $\sqrt{d_k}$:缩放,防止内积过大导致 softmax 饱和、梯度消失;
- **多头(Multi-Head)**:并行多组 Q/K/V,让模型在不同子空间关注不同关系。

**关键:因果掩码(Causal Mask)**。GPT 在 softmax 前把"未来"位置的分数置为 $-\infty$,保证 $w_t$ 看不到 $w_{\ge t}$:

```
注意力分数矩阵 QK^T (T×T),上三角 mask 为 -inf 后 softmax:
          w1   w2   w3   w4
   w1  [  ✓    ✗    ✗    ✗ ]   # w1 只能看自己
   w2  [  ✓    ✓    ✗    ✗ ]
   w3  [  ✓    ✓    ✓    ✗ ]
   w4  [  ✓    ✓    ✓    ✓ ]
```

### 6.4 GPT 整体架构(decoder-only)

```
text
 │  tokenizer (BPE, ch2)         # 子词分词,解决 OOV、压缩词表
 ▼
token ids
 │  Token Embedding + Positional Embedding(ch2~ch3)
 ▼
┌───────────────────────────────────────┐
│  Transformer Block × N (ch3~ch4)      │
│   ┌─────────────────────────────┐    │
│   │  Masked Multi-Head Attention │    │
│   │  + Residual + LayerNorm      │    │
│   ├─────────────────────────────┤    │
│   │  Feed-Forward (MLP)          │    │
│   │  + Residual + LayerNorm      │    │
│   └─────────────────────────────┘    │
└───────────────────────────────────────┘
 │
 ▼
LM Head: 线性层 → softmax over 词表   # 输出下一个词概率(ch4)
```

- **分词(BPE, ch2)**:把词拆成子词,既控制词表大小又避免 OOV —— 这步直接对应 GPT-2 目录里的 `the-verdict.txt` 分词实验;
- **Token + Position Embedding(ch2)**:词向量 + 位置向量(Transformer 本身无位置感知);
- **N 个 Transformer Block(ch3~ch4)**:masked self-attention + FFN,带残差和 LayerNorm;
- **LM Head(ch4)**:映射回词表维度,做下一个词预测;
- **训练(ch5)**+ **加载预训练 GPT-2 124M 权重微调分类/生成(ch5~ch7)**。

### 6.5 预训练 → 迁移的范式

```
阶段1  Pretrain:海量无标注文本,自监督预测下一个词 → 学到通用语言表示
阶段2  Fine-tune / Prompt:少量下游任务数据微调(分类),或直接 zero/few-shot 提示
```

**GPT-2 的关键观察**:当模型规模足够大、预训练足够充分,**零样本(zero-shot)**能力涌现 —— 不微调,直接用提示完成翻译、摘要、问答。GPT-3 进一步把它推向 **few-shot in-context learning**。

> 一句话:**GPT = "预测下一个词"这个 n-gram 时代就有的目标 + Transformer 的可并行长程建模 + 大规模自监督预训练。** 它把"表示方法"统一进了一个端到端模型:不再单独学词向量、不再分预训练表示和下游网络,词义、上下文、甚至任务能力,全在一次预训练里涌现。

---

## 7. 全景总结

```
one-hot        : 离散身份编号,无语义
   │  问题:无语义、维度爆炸
   ▼
TF-IDF         : 词袋加权,知道哪些词重要,仍无语义、丢语序
   │  问题:统计计数,无泛化
   ▼
n-gram LM      : 第一次做"预测下一个词",但 O(|V|^n) 爆炸、长程弱
   │  问题:离散符号,需要连续向量
   ▼
word2vec       : 稠密分布式向量,有语义、有类比关系 —— 但静态
   │  问题:一词一向量,无法消歧
   ▼
ELMo           : 上下文相关(biLSTM),消歧成功 —— 但 LSTM 难并行
   │  问题:LSTM 是天花板
   ▼
GPT            : Transformer + 自回归预训练,端到端,可并行、长程强、能力随规模涌现
```

**贯穿始终的一条主线**:目标函数其实没怎么变 —— 从 n-gram 到 GPT,**都是"根据前文预测下一个词"**。
变的是**用什么去算这个预测**:计数统计 → 浅层神经网络(lookup 词向量)→ 深层循环网络(biLSTM)→ 深层注意力网络(Transformer)。表示方法的演进,本质是**模型容量的演进 + 表示的统一**:从"手工设计的稀疏特征"一步步走向"一个端到端模型学出一切"。

---

### 参考资料
- Mikolov et al., 2013. *Distributed Representations of Words and Phrases and their Compositionality* (word2vec / 负采样).
- Peters et al., 2018. *Deep contextualized word representations* (ELMo).
- Radford et al., 2018/2019. *Improving Language Understanding by Generative Pre-Training* / *Language Models are Unsupervised Multitask Learners* (GPT-1/2).
- Vaswani et al., 2017. *Attention Is All You Need* (Transformer).
- Raschka, 2024. *Build a Large Language Model (From Scratch)* (本仓库 ch2~ch7 配套).
