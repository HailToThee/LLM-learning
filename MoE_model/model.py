import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.scale

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Router(nn.Module):
    def __init__(self, dim, num_experts, k=2):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.linear = nn.Linear(dim, num_experts)

    def forward(self, x):
        logits = self.linear(x)  # (batch_size, seq_len, num_experts)
        topk_logits, topk_indices = torch.topk(logits, self.k, dim=-1)
        return logits, topk_logits, topk_indices

class MoEBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, k=2, dropout=0.1):
        super(MoEBlock, self).__init__()
        self.router = Router(dim, num_experts, k)
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim, dropout) for _ in range(num_experts)])
        self.norm = RMSNorm(dim)
        self.k = k
        self.num_experts = num_experts

    def compute_router_z_loss(self, logits, topk_indices):
        batch_size, seq_len, num_experts = logits.shape
        router_probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, num_experts)

        expert_mask = torch.zeros_like(logits)
        for i in range(batch_size):
            for j in range(seq_len):
                for k_idx in range(self.k):
                    expert_idx = topk_indices[i, j, k_idx]
                    expert_mask[i, j, expert_idx] = 1.0

        dispatcher = torch.softmax(logits, dim=-1)
        auxiliary_loss = torch.mean(
            torch.sum(dispatcher * expert_mask, dim=(0, 1)) ** 2
        ) * self.num_experts
        
        return auxiliary_loss

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        logits, topk_logits, topk_indices = self.router(x)
        
        expert_outputs = torch.zeros_like(x)
        
        router_z_loss = self.compute_router_z_loss(logits, topk_indices)
        
        for i in range(batch_size):
            for j in range(seq_len):
                for k_idx in range(self.k):
                    expert_idx = topk_indices[i, j, k_idx].item()
                    expert_weight = torch.softmax(topk_logits[i, j], dim=-1)[k_idx]
                    expert_output = self.experts[expert_idx](x[i:i+1, j:j+1, :])
                    expert_outputs[i, j, :] += expert_weight * expert_output[0, 0, :]
        
        return self.norm(expert_outputs), router_z_loss
    
class MoEModel(nn.Module):
    def __init__(self, vocab_size, dim, hidden_dim, num_experts, num_layers, k=2, dropout=0.1):
        super(MoEModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            MoEBlock(dim, hidden_dim, num_experts, k, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, return_aux_loss=True):
        x = self.embedding(input_ids)
        total_aux_loss = 0.0
        
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
        
        x = self.norm(x)
        logits = self.output(x)
        
        if return_aux_loss:
            return logits, total_aux_loss
        else:
            return logits
    
