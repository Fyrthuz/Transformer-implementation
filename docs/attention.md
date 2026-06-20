# Attention Mechanisms

All 9 attention modules follow the same interface:
```python
class AttentionModule(nn.Module):
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor
```
- `x`: `(B, T, d_model)`
- `mask`: `(B, 1, T, T)` or `(B, T, T)` — causal mask with 1s where attention is allowed
- Returns: `(B, T, d_model)`

They are registered via `@register_attention("name")` and instantiated through `get_attention(name, **kwargs)`.

---

### 1. MHA — Multi-Head Attention (`@register_attention("mha")`)

**Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Standard dot-product attention with separate Q/K/V projections for each head:

```
Q = x @ W_q    → (B, T, d_model)
K = x @ W_k    → (B, T, d_model)
V = x @ W_v    → (B, T, d_model)

Reshape to (B, n_heads, T, d_k)
scores = Q @ K^T / sqrt(d_k)
scores[mask == 0] = -inf
attn = softmax(scores)
context = attn @ V
output = concat_heads(context) @ W_o
```

- `d_k = d_model / num_heads`
- All heads have equal size.
- Bias-free projections.

---

### 2. MQA — Multi-Query Attention (`@register_attention("mqa")`)

**Paper:** [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

All query heads share a single key-value head:

```
Q: (B, n_heads, T, d_k)   # full multi-head
K: (B, 1, T, d_k)          # shared single head
V: (B, 1, T, d_k)          # shared single head
```

- Dramatically reduces KV-cache memory at inference time.
- Attention computation broadcasts the single KV head across all query heads.

---

### 3. GQA — Grouped-Query Attention (`@register_attention("gqa")`)

**Paper:** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

Intermediate between MHA and MQA: `num_kv_heads` groups of query heads share one KV head:

```
Q: (B, n_heads, T, d_k)
K: (B, n_kv_heads, T, d_k)     # n_kv_heads < n_heads
V: (B, n_kv_heads, T, d_k)

Repeat K/V by factor n_heads / n_kv_heads → (B, n_heads, T, d_k)
```

- Default: `num_kv_heads = max(1, num_heads // 4)` if not explicitly set.
- Balance between MHA quality and MQA efficiency.

---

### 4. Linear Attention (`@register_attention("linear")`)

**Paper:** [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)

Replaces softmax attention with a linear kernel: `sim(Q, K) = ELU(Q) + 1` applied elementwise:

```
Q' = ELU(Q) + 1
K' = ELU(K) + 1

Without mask:    context = Q' @ (K'^T @ V)
With mask (autoregressive, cumulative sum scan):
    for t in 1..T:
        KV_cumsum += K'[t]^T @ V[t]
        out[t] = Q'[t] @ KV_cumsum
```

- **Complexity**: O(T d²) instead of O(T² d) for standard attention.
- The ELU feature map ensures non-negative similarities (no negative attention).
- The causal scan mode (used during training with mask) is sequential O(T) and slower for short sequences but scales better for long ones.

---

### 5. Window Attention (`@register_attention("window")`)

**Paper:** [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

Each token attends only to tokens within a sliding window of size `window_size`:

```
half_w = window_size // 2
for each position i:
    attend to j in [i - half_w, i + half_w]
```

- A boolean mask zeros out scores outside the window before softmax.
- Causal mask is applied on top of the window mask.

---

### 6. Dilated Attention (`@register_attention("dilated")`)

**Paper:** [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (dilated sliding window variant)

Like window attention but with dilation: within the window, attends only to every `dilation`-th position:

```
half_w = window_size // 2
for each position i:
    for j in [i - half_w*dilation, i + half_w*dilation]:
        include if (j - i) % dilation == 0 or |j - i| ≤ 2
```

- The `|j-i| ≤ 2` exception ensures immediate neighbors are always accessible regardless of dilation.
- Dilated windows give broader context coverage for the same computational cost.

---

### 7. Global-Local Attention (`@register_attention("global_local")`)

**Paper:** [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (global + sliding window)

Combines learned global tokens with local windowed attention for each token:

```
G learned global tokens, shape (G, d_model)

x_cat = [global_tokens; x]    # (B, G+T, d_model)

Global tokens attend to all positions (full mask).
Local tokens attend to global tokens + local window.
```

- `num_global_tokens` (default: 8) are learnable parameters added to input.
- Global tokens allow every local token to incorporate task-level context.
- The causal mask is extended with ones in the global token region.

---

### 8. Mamba Block (`@register_attention("mamba")`)

**Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

A selective state-space model that replaces attention entirely:

```
1. Project input: [x_proj; z] = x @ W_in    (d_model → 2 * d_inner)
2. Conv1d on x_proj (depthwise, kernel=d_conv)
3. SiLU activation on x_proj
4. Compute Δ, B, C from x_proj:
   Δ = softplus(x_proj @ W_dt)
   A = -exp(log A)  (learned log-space A)
5. Selective scan (sequential over T):
   h_t = exp(Δt * A) * h_{t-1} + Δt * B_t * x_proj_t
   y_t = h_t @ C_t + D * x_proj_t
6. Gate: y = y * SiLU(z)
7. Output projection: y @ W_out
```

**Configuration:**
| Param | Default | Description |
|-------|---------|-------------|
| `d_state` | 16 | State dimension per channel |
| `expand_factor` | 2 | Inner dimension = d_model × expand_factor |
| `d_conv` | 4 | Conv1d kernel size |

**Key details:**
- `A` is parameterized in log-space (`A_log`) and negated on-the-fly to ensure stability: `A = -exp(A_log)`.
- The selective scan is the core O(T) sequential loop that makes Mamba linear in sequence length.
- `B` and `C` are input-dependent (selective), allowing the model to "select" what to store or recall.

---

### 9. SSM Block (`@register_attention("ssm")`)

A simplified state-space model variant (no convolution, no input-dependent B/C):

```
1. Project: [x_proj; z] = x @ W_in    (d_model → 2*d_model)
2. SiLU activation on x_proj
3. Fixed Δ = softplus(log_dt)  (learned scalar per channel)
4. Fixed B, C: learnable parameters (not input-dependent)
5. Same scan as Mamba but with static A, B, C, Δ
6. Gate: y = y * SiLU(z)
```

**Key differences from Mamba:**
- No depthwise convolution.
- `B`, `C`, `Δ` are static learned parameters (not functions of input).
- `d_state` applies per channel of `d_model` (not `d_inner`).
- Much simpler and fewer parameters than Mamba, but less expressive.
