"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

"""Lab 2: scaled dot-product attention and multi-head attention."""

import math
import torch


def attention(q, k, v, mask=None):
    """Compute scaled dot-product attention."""

    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return torch.matmul(weights, v)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        output = attention(q, k, v, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        return self.out_proj(output)
