# Positional Encodings

All 3 position encoding modules are registered via `@register_position("name")` and instantiated through `get_position(name, d_model, dropout, max_len)`. Interface:

```python
class PositionModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor
```

Each receives token embeddings `(B, T, d_model)` and returns position-aware embeddings of the same shape. All apply dropout to the output.

---

### 1. Sinusoidal Encoding (`@register_position("sinusoidal")`)

**Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Fixed (non-learned) sinusoidal positional encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Implementation:
```python
pe = torch.zeros(max_len, d_model)
position = arange(max_len).unsqueeze(1)                                # (max_len, 1)
div_term = exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))     # (d_model/2)
pe[:, 0::2] = sin(position * div_term)
pe[:, 1::2] = cos(position * div_term)
```

- Pre-computed table of size `(max_len, d_model)`, registered as a buffer (not a parameter).
- Added to token embeddings via broadcasting: `x + pe[:T]`.
- The frequency decreases along the channel dimension, creating a multi-resolution encoding.

---

### 2. RoPE — Rotary Position Embedding (`@register_position("rope")`)

**Paper:** [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

Rotates query and key embeddings by an angle proportional to position, encoding relative position through the dot product:

```
For position t, dimension pair (2i, 2i+1):
    θ_i = 1 / theta_rope^(2i/d_model)
    rotation matrix = [[cos(t*θ_i), -sin(t*θ_i)],
                       [sin(t*θ_i),  cos(t*θ_i)]]
```

Implementation (applied to full embeddings, not just Q/K):
```python
half = d_model // 2
x1 = x[..., :half]
x2 = x[..., half:]
cos = cos(positions @ inv_freq)
sin = sin(positions @ inv_freq)
rotated = cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

**Configuration:**
| Param | Default | Description |
|-------|---------|-------------|
| `rope_theta` | 10000.0 | Base frequency for rotary angles |
| `max_len` | 5000 | Maximum sequence length (for cache pre-allocation) |

Two implementations exist in the file:
- `RotaryEmbedding` — computes cos/sin fresh each forward pass.
- `RotaryEmbeddingSimple` — caches cos/sin up to `max_len` to avoid recomputation.

Both are registered under `"rope"`; the last one registered (`RotaryEmbeddingSimple`) is the active one.

---

### 3. None (`@register_position("none")`)

No positional encoding — identity operation:

```python
def forward(self, x):
    return dropout(x)
```

- Useful when the attention mechanism already encodes position (e.g., Mamba/SSM which are inherently sequential).
- On TinyShakespeare character-level, "none" outperforms sinusoidal and RoPE when paired with Mamba, since the SSM's sequential scan already captures order.
