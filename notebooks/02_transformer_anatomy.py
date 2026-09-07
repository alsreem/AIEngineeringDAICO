"""Lab 2: Anatomy of a Transformer."""

import math

import torch
import torch.nn.functional as F

from bayan.attention import attention, MultiHeadAttention


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("Bayan Lab 2 — Transformer Anatomy")
    print("=" * 70)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    ours = attention(q, k, v)
    pytorch = F.scaled_dot_product_attention(q, k, v)

    max_diff = (ours - pytorch).abs().max().item()

    print("\n1) Attention numerical equivalence")
    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Matches PyTorch (atol=1e-6): {torch.allclose(ours, pytorch, atol=1e-6)}")

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    weights = torch.softmax(scores, dim=-1)

    print("\n2) Attention weights")
    print("Shape:", tuple(weights.shape))
    print("Head 0:")
    print(weights[0, 0])

    print("\n3) Multi-Head Attention")

    x = torch.randn(1, 4, 8)
    mha = MultiHeadAttention(d_model=8, num_heads=2)

    output = mha(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(output.shape))

    print("\n4) Causal mask")

    seq_len = 4

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )

    print(causal_mask.int())

    q2 = torch.randn(1, 1, seq_len, 8)
    k2 = torch.randn(1, 1, seq_len, 8)
    v2 = torch.randn(1, 1, seq_len, 8)

    causal_output = attention(q2, k2, v2, mask=causal_mask)

    causal_scores = torch.matmul(
        q2, k2.transpose(-2, -1)
    ) / math.sqrt(q2.size(-1))

    causal_scores = causal_scores.masked_fill(
        ~causal_mask, float("-inf")
    )

    causal_weights = torch.softmax(causal_scores, dim=-1)

    print("Causal attention weights:")
    print(causal_weights[0, 0])

    print("Future positions blocked:",
          torch.all(causal_weights[0, 0].triu(1) == 0).item())

    print("\n5) PAD attention leakage")

    pad_mask = torch.tensor(
        [[1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )

    q3 = torch.randn(1, 1, 5, 8)
    k3 = torch.randn(1, 1, 5, 8)
    v3 = torch.randn(1, 1, 5, 8)

    scores3 = torch.matmul(
        q3, k3.transpose(-2, -1)
    ) / math.sqrt(q3.size(-1))

    
    weights_without_mask = torch.softmax(scores3, dim=-1)

    
    scores_with_mask = scores3.masked_fill(
        ~pad_mask[:, None, None, :],
        float("-inf"),
    )

    weights_with_mask = torch.softmax(
        scores_with_mask,
        dim=-1,
    )

    pad_positions = ~pad_mask

    pad_mass_without = weights_without_mask[
        ..., pad_positions[0]
    ].sum().item()

    pad_mass_with = weights_with_mask[
        ..., pad_positions[0]
    ].sum().item()

    print(f"PAD attention mass without mask: {pad_mass_without:.6f}")
    print(f"PAD attention mass with mask   : {pad_mass_with:.6f}")

    print("\nLab 2 anatomy checks complete.")


if __name__ == "__main__":
    main()

