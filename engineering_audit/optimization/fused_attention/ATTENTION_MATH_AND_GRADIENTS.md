# ForeSWE geo-biased attention: exact mathematics and gradients

Source: `fore_swe/custom_encoder_model.py`, `MultiHeadAttention` (unchanged upstream code), used by all
8 main-encoder layers (`EncoderLayer.self_attn`, d_model = 3072, 16 heads).

## Forward, exactly as implemented

```python
attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)       # [B, H, N, N]
if mask is not None: attn_scores = attn_scores.masked_fill(mask == 0, -1e9)       # never used: mask=None
attn_scores = torch.softmax(attn_scores, dim=-1)                                  # softmax #1
distances   = 1 / (1 + distances)                                                 # [N, N]
angularities = 1 / (1 + angularities)                                             # [N, N]
new_score = self.alpha * attn_scores + self.beta * distances + self.gamma * angularities
modified_attn_weights = torch.softmax(new_score, dim=-1)                          # softmax #2
output = torch.matmul(modified_attn_weights, V)                                   # [B, H, N, d_k]
```

| Symbol | Shape (production) | Dtype | Trainable | Notes |
|---|---|---|---|---|
| Q, K, V | [1, 16, 512, 192] | fp32 (bf16 under autocast) | via W_q, W_k, W_v | `split_heads` returns a **transposed view** (non-contiguous) |
| d_k | 192 = 3072 / 16 | – | – | scale 1/√192 applied by division |
| dist | [512, 512] | fp32 | no | Haversine distance between stations (km), ≥ 0 |
| ang | [512, 512] | fp32 | no | angularity between station vectors (degrees, now Kahan 2·atan2), ≥ 0 |
| α, β, γ | [1] each | fp32 | **yes**, one set per layer, shared by all heads | initialised `randn(1)`. Trained Plan B values: layer 0 (0.31, −0.21, 0.20), layer 7 (−0.55, 0.41, −0.02) |
| mask | – | – | – | always `None` in ForeSWE |

Broadcasting: the priors `1/(1+dist)` and `1/(1+ang)` are [N, N], broadcast over batch and heads. Both
softmaxes normalise over the **last axis (keys)**. The prior transforms are recomputed in every layer and
every step even though dist and ang are constant.

Mathematically, with D = 1/(1+dist) and A = 1/(1+ang):

    S  = Q Kᵀ / √d_k
    P₁ = softmax_row(S)
    Z  = α P₁ + β D + γ A
    P₂ = softmax_row(Z)
    O  = P₂ V

### Not FlashAttention-equivalent

Standard (Flash) attention computes softmax(S + bias)·V. Here the bias is added **after** a first
normalisation and the result is renormalised. softmax(α·softmax(S) + βD + γA) cannot be rewritten as
softmax(S' + bias) for any S' that is linear in QKᵀ: α·softmax(S) is a non-linear (row-normalised)
function of S. `F.scaled_dot_product_attention` therefore cannot implement this operator, and the
two-softmax form is preserved.

### Bound that enables single-pass softmax #2

P₁ ∈ [0, 1], D ∈ (0, 1] and A ∈ (0, 1] because dist, ang ≥ 0. Hence
Z ≤ |α| + |β| + |γ| =: c. softmax is shift-invariant, so softmax(Z) = exp(Z − c) / Σ exp(Z − c) **exactly**,
with exp(Z − c) ∈ (e^(−2c), 1]: no overflow, and no row maximum is needed. This removes the online-max
rescaling that FlashAttention needs for the second softmax. Softmax #1 still needs its row max and sum
(m₁, l₁) before P₁ is known, which forces **two passes** over the keys in the forward.

## Backward (all derived for the implementation; verified by `torch.autograd.gradcheck` in float64)

Given dO = ∂L/∂O:

    dV  = P₂ᵀ dO
    dP₂ = dO Vᵀ
    δ₂  = rowsum(dP₂ ∘ P₂) = rowsum(dO ∘ O)                     (FlashAttention identity)
    dZ  = P₂ ∘ (dP₂ − δ₂)                                        (softmax #2 backward)
    dα  = Σ dZ ∘ P₁,   dβ = Σ dZ ∘ D,   dγ = Σ dZ ∘ A             (sums over B, H, rows, cols)
    dP₁ = α dZ
    δ₁  = rowsum(dP₁ ∘ P₁) = α · rowsum(dZ ∘ P₁)
    dS  = P₁ ∘ (dP₁ − δ₁)                                        (softmax #1 backward)
    dQ  = dS K / √d_k,    dK = dSᵀ Q / √d_k
    d(dist), d(ang): not required (no grad; constant inputs)

Dependency structure:
- δ₂ needs only O and dO (cheap, elementwise).
- δ₁ needs a **full pass over all keys of a row** (it depends on dZ, which depends on P₂ and dP₂), so dS,
  and therefore dQ and dK, need a second pass after δ₁ is known.
- dα, dβ, dγ are global reductions (one scalar each per layer).

This is the structural difference from FlashAttention's backward, which needs only δ = rowsum(dO ∘ O)
before a single key-block loop.

## Intermediate tensors in the eager implementation (per layer, per step, fp32)

S, P₁ (softmax #1), D and A (recomputed [N, N]), αP₁ + βD + γA (two temporaries), P₂: each
[1, 16, 512, 512] = **16 MiB** (D and A are 1 MiB, broadcast). Autograd saves P₁, P₂ and the inputs:
measured **34 MiB saved for backward** and **76 MiB peak extra** per call (TF32,
`benchmark_results/attention_study.json`). Eager launches **16 kernels forward** and **40 forward +
backward**.
