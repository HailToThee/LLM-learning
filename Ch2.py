# %%
import re
with open('the-verdict.txt', 'r') as f:
    raw_text = f.read()
print("Total characters:", len(raw_text))
print(raw_text[:100])

# %%
text="Hello world. This is a new text."
result=re.split(r'([,.]|    \s)',text)
result=[item for item in result if item.strip()]
print(result)

# %% [markdown]
# `item.strip()` 的意思是去除字符串 `item` 两端的空白字符（包括空格、换行、制表符等），返回处理后的新字符串。常用于清理输入或分割后的数据。
# 
# 在`.split`方法中，参数`r''`表示**原始字符串**（raw string）。
# 加上`r`前缀后，字符串中的反斜杠`\`不会被当作转义字符处理，而是原样保留。这样可以方便地编写正则表达式，比如`\s`表示匹配空白字符。
# 
# 例如：
# - `r'\s'`：匹配空白字符（空格、制表符等）
# - `'\n'`：普通字符串，`\n`会被解析为换行符
# 
# 使用`r''`可以避免转义带来的混淆，尤其在正则表达式中非常常见。

# %%
preprocessed = re.split(r'([,.:?!;_"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(preprocessed[:30])
#对raw_text进行预处理，使用正则表达式将文本分割成单词和标点符号，并去除空白字符。

# %%
all_words = sorted(set(preprocessed))
vocal_size= len(all_words)
print(vocal_size)

vocab = {token:i for i, token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    if i < 10:
        print(f"{item[0]}: {item[1]}")
#创建一个词汇表，将所有唯一的单词和标点符号映射到一个唯一的索引。

# %%
class SimpleTokenizer:
    def __init__(self, vocab):
        self.stoi = vocab
        self.itos = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:?!;_"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        ids = [self.stoi[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.itos[i] for i in ids)
        text = re.sub(r'\s+([,.:?!;_"()\'])', r'\1', text)
        return text
tokenizer = SimpleTokenizer(vocab)
text = """""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Encoded IDs:", ids)
print("Decoded text:", tokenizer.decode(ids))

# text = """hello world. This is a new text."""
# ids = tokenizer.encode(text)
# print("Encoded IDs:", ids)
# print("Decoded text:", tokenizer.decode(ids))

# %% [markdown]
# 这段代码实现了一个简单的分词器（`SimpleTokenizer`），用于将文本分割成单词和标点，并将它们与唯一索引进行映射。具体说明如下：
# 
# ---
# 
# ### 词汇表创建
# 
# - 首先，`vocab` 是一个字典，将所有唯一的单词和标点符号映射到唯一的索引（整数）。
# - 例如：`{'hello': 0, 'world': 1, '.': 2, ...}`
# 
# ### SimpleTokenizer 类
# 
# - `__init__` 方法：
#   - `self.stoi` 保存字符串到索引的映射（String to Index）。
#   - `self.itos` 保存索引到字符串的反向映射（Index to String）。
# 
# - `encode(text)` 方法：
#   - 使用正则表达式将文本分割成单词和标点符号。
#   - 去除空白字符。
#   - 将每个分割后的 token 映射为对应的索引，返回索引列表。
# 
# - `decode(ids)` 方法：
#   - 根据索引列表还原为字符串。
#   - 用正则表达式处理空格，使标点符号格式更自然。
# 
# ### 使用示例
# - 创建分词器对象：`tokenizer = SimpleTokenizer(vocab)`
# - 编码文本为索引：`ids = tokenizer.encode(text)`
# - 解码索引为文本：`tokenizer.decode(ids)`
# -
# 但是由于未对不存在的 token 进行处理，可能会导致编码时出现错误。

# %%
# 构建词汇表并添加特殊 token
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.stoi = vocab
        self.itos = {i: s for s, i in vocab.items()}  # 修正键值顺序

    def encode(self, text):
        preprocessed = re.split(r'([,.:?!;_"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        ids = []
        for token in preprocessed:
            if token in self.stoi:
                ids.append(self.stoi[token])
            else:
                ids.append(self.stoi["<|unk|>"])  # 未知 token
        return ids

    def decode(self, ids):
        text = " ".join(self.itos.get(i, "<|unk|>") for i in ids)
        text = re.sub(r'\s+([,.:?!;_"()\'])', r'\1', text)
        return text

text1 = "Hello world. This is a new text."
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# %%
##基于BPE的分词器
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text1 = "Hello world. This is a new text."
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
integers = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))

text1 = "Akwirw ier"
integers = tokenizer.encode(text1,allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))

# %%
with open('the-verdict.txt', 'r',encoding='utf-8') as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
content_size=4
x = enc_sample[:content_size]
y = enc_sample[1:content_size+1]
print(f"x:{x}")
print(f"y:{y}")

# %%




