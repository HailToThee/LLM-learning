# MoE understanding
Mixture of Experts(MoE) is a subtitude of dense layers(normally feed forward layers in transformers).

Full parameters will be loaded on memory, but only a part of parameters will be activated during inference, which can save memory and computation cost.
think of this situation.
```
a model claim to have 8*7B parameters. normally, context_length is set to 4096.

Less than 2*7B parameters will be activated during inference, which can save memory and computation cost.
```

![alt text](./mixture-of-experts/moe.png)

```
given an input
x = [x1, x2, x3, x4]
embedding:
x = nn.Embedding+PositionEmbedding(x)
x = [x1, x2, x3, x4] -> [e1, e2, e3, e4]
To transformers:
x = transformer(x)
x = [e1, e2, e3, e4] -> [t1, t2, t3, t4] # b*l*d

router H:
posibility = router(x) (b*l*num_experts)

raw_gates = posibility.softmax(dim=-1) # b*l*num_experts
gated_1, index_1 = top1(raw_gates) -> expert_id (top1)
gated_2, index_2 = top1(raw_gates.remove(index_1)) -> expert_id (top2)

mask_1 = F.onehot(index_1, num_experts) #punish the top1
desity_1 = mask_1.mean(dim=-2)  #equal to divide by batch_size, b*num_experts
!--important
auxliary loss
density_1_proxy = raw_gates.mean(dim=-2) b*num_experts

auxilairy_loss = (density_1_proxy * density_1).mean()* float(num_experts ** 2)
```