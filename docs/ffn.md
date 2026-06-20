# Feed-Forward Networks

All 4 FFN variants are registered via `@register_ffn("name")` and instantiated through `get_ffn(name, **kwargs)`. Interface:

```python
class FFNModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor
    def auxiliary_losses(self) -> list[Tensor]  # optional
```

All variants operate on `(B, T, d_model)` and return `(B, T, d_model)`.

---

### 1. Standard FFN (`@register_ffn("standard")`)

**Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The classic two-layer MLP with a configurable activation:

```
x → Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model) → Dropout
```

**Activation options:** `gelu` (default), `relu`, `silu`.

---

### 2. SwiGLU FFN (`@register_ffn("swiglu")`)

**Paper:** [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

Gated linear unit variant using SiLU (Swish) activation:

```
x → w1(x) → SiLU → element-wise multiply with w3(x) → w2 → Dropout
```

```python
return w2(SiLU(w1(x)) * w3(x))
```

- Three weight matrices: `w1`, `w3` project `d_model → d_ff`, `w2` projects back.
- The gate mechanism (`SiLU(w1(x)) * w3(x)`) provides an adaptive non-linearity.
- Bias-free projections.
- Note: `d_ff` in SwiGLU is used as-is (no 2/3 scaling factor applied to hidden dimension).

---

### 3. Gated FFN (`@register_ffn("gated")`)

Sigmoid-gated variant with separate gate path:

```
x → w1(x) → GELU → multiply with sigmoid(w_gate(x)) → w2 → Dropout
```

```python
gate = sigmoid(w_gate(x))
hidden = GELU(w1(x))
return w2(hidden * gate)
```

- Three weight matrices: `w1`, `w_gate` (gate), and `w2`.
- Sigmoid gate provides a soft gating mechanism.
- Similar to SwiGLU but uses GELU + sigmoid instead of SiLU + identity.

---

### 4. Mixture-of-Experts (`@register_ffn("moe")`)

**Paper:** [Mixture of Experts Explained](https://arxiv.org/abs/2101.03961) / [Switch Transformers](https://arxiv.org/abs/2101.03961)

Sparsely-gated mixture of `num_experts` expert FFNs, each identical to a StandardFFN:

```
x_flat → Router(x_flat) → top-k routing → weighted expert outputs
```

**Forward pass:**
```python
x_flat: (B*T, d_model)
logits = Router(x_flat)                                  # (B*T, num_experts)
weights, indices = topk(logits, k=top_k)                 # → top-k routing
weights = softmax(weights, dim=-1)

for each expert e:
    mask = (indices[:, i] == e) for each top-k slot
    out[mask] += weights[mask] * expert_e(x_flat[mask])

return out.view(B, T, d_model)
```

**Load-balancing loss:**
```python
frac = count of tokens per expert / total tokens
router_probs = mean(softmax(logits), dim=0)
loss = num_experts * sum(frac * router_probs)
```
- Minimized when all experts receive equal probability mass.
- Weighted by `training.loss.moe_load_balance_coef` (default: 0.01) and added to the total loss.

**Configuration:**
| Param | Default | Description |
|-------|---------|-------------|
| `num_experts` | 8 | Number of expert sub-networks |
| `top_k` | 2 | Number of experts activated per token |

**Auxiliary losses:** Exposed via `auxiliary_losses()` → collected by `TransformerBlock.auxiliary_losses()` → summed in training loop.
