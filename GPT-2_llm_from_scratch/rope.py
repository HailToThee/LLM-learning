import numpy as np
import torch

# freq = 10000 ** -(torch.arange(0, dim, 2) / dim)
def get_rotary_frequencies(dim: int, seq_len: int, base: int = 10000):
    """
    Get the rotary frequencies for a given dimension and sequence length.
    Args:
        dim (int): The dimension of the model.
        seq_len (int): The maximum sequence length.
        base (int): The base frequency. Default is 10000.
    """
    # theta_i = 10000 ^ (-2i/d) i = 0, 1, ..., d/2-1
    i = torch.arange(0, dim//2, dtype=torch.float32)
    freqs = base ** -(2*i / dim)
    position = torch.arange(seq_len, dtype=torch.float32)
    # freqs shape: (d/2,), position shape: (seq_len,)
    angles = torch.outer(position, freqs)  # shape: (seq_len, d/2)
    return angles

seq_len = 128
dim = 64
rotary_frequencies = get_rotary_frequencies(dim, seq_len)
print(rotary_frequencies.shape)  # should be (seq_len, dim/2)
print(rotary_frequencies[0][:5])
print(rotary_frequencies[1][:5])
print(rotary_frequencies[2][:5])
# euler's formula: e^(ix) = cos(x) + i*sin(x), e^(-ix) = cos(x) - i*sin(x)

def get_rotary_embedding(angles: torch.Tensor):
    """
    Get the rotary embedding from the angles.
    Args:
        angles (torch.Tensor): The angles for the rotary embedding. Shape: (seq_len, dim/2)"""
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    # Combine the cosine and sine values to create the rotary embedding
    cos = torch.cat([cos_angles, cos_angles], dim=-1)  # shape: (seq_len, dim)
    sin = torch.cat([sin_angles, sin_angles], dim=-1)  # shape: (seq_len, dim)
    return cos, sin

def rotate_half(x: torch.Tensor):
    """
    although in rope, out should be: x1_theta1 - x2_theta1, x2_theta1 - x1_theta1.
    however, pair doesnot matter if we use <x1, x3>, <x2, x4> or <x1, x2>, <x3, x4>.
    Rotate the input tensor by half of its dimensions.
    Args:
        x (torch.Tensor): The input tensor. Shape: (batch_size, seq_len, dim)"""
    dim = x.shape[-1]
    assert dim % 2 == 0, "Dimension must be even for rotary embedding."
    x1 = x[..., :dim//2]  # shape: (batch_size, seq_len, dim/2)
    x2 = x[..., dim//2:]  # shape: (batch_size, seq_len, dim/2)
    return torch.cat([-x2, x1], dim=-1)  # shape: (batch_size, seq_len, dim)

x = torch.tensor([1,2,3,4,5,6]) #[theta1, theta2, theta3, theta1, theta2, theta3]
print(rotate_half(x))  # should be [-4, -5, -6, 1, 2, 3]

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply the rotary positional embedding to the query and key tensors.
    Args:
        q (torch.Tensor): The query tensor. Shape: (batch_size, seq_len, dim)
        k (torch.Tensor): The key tensor. Shape: (batch_size, seq_len, dim)
        cos (torch.Tensor): The cosine values for the rotary embedding. Shape: (1, seq_len, dim)
        sin (torch.Tensor): The sine values for the rotary embedding. Shape: (1, seq_len, dim)"""
    print(cos)
    print(sin)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

q = torch.tensor([[[1,2,3,4,5,6], [7,8,9,10,11,12]], [[13,14,15,16,17,18], [19,20,21,22,23,24]]], dtype=torch.float32) # shape: (2, 2, 6)
k = q.clone()
angles = get_rotary_frequencies(dim=6, seq_len=2)  # shape: (2, 3)
cos, sin = get_rotary_embedding(angles)  # shape: (2, 6)
cos = cos.unsqueeze(0)  # shape: (1, 2, 6)
sin = sin.unsqueeze(0)  # shape: (1, 2, 6)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

print(f"q_rot.shape: {q_rot.shape}")
print(f"k_rot.shape: {k_rot.shape}")
print(q_rot)
print(k_rot)


def verify_relative_position_invariance():
    """
    验证 RoPE 的相对位置不变性
    """
    dim = 64
    max_seq_len = 100

    # 预计算 cos/sin
    angles = get_rotary_frequencies(dim, max_seq_len)
    cos, sin = get_rotary_embedding(angles)

    # 创建两个相同的向量
    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # 场景 1：q 在位置 0，k 在位置 5（相对位置 = 5）
    cos1_q, sin1_q = cos[0:1], sin[0:1]
    cos1_k, sin1_k = cos[5:6], sin[5:6]

    q1_rot, _ = apply_rotary_pos_emb(q, q, cos1_q, sin1_q)
    _, k1_rot = apply_rotary_pos_emb(k, k, cos1_k, sin1_k)

    dot_product_1 = (q1_rot * k1_rot).sum()

    # 场景 2：q 在位置 10，k 在位置 15（相对位置仍然是 5）
    cos2_q, sin2_q = cos[10:11], sin[10:11]
    cos2_k, sin2_k = cos[15:16], sin[15:16]

    q2_rot, _ = apply_rotary_pos_emb(q, q, cos2_q, sin2_q)
    _, k2_rot = apply_rotary_pos_emb(k, k, cos2_k, sin2_k)

    dot_product_2 = (q2_rot * k2_rot).sum()

    print(f"位置 (0, 5) 的内积: {dot_product_1.item():.6f}")
    print(f"位置 (10, 15) 的内积: {dot_product_2.item():.6f}")
    print(f"差异: {abs(dot_product_1.item() - dot_product_2.item()):.10f}")
    print("验证通过！" if abs(dot_product_1.item() - dot_product_2.item()) < 1e-5 else "验证失败！")


verify_relative_position_invariance()
