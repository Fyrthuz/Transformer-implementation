# transformer.py
import torch
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Proyección lineal y división en cabezales
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Forzamos a que combine perfecto con las dimensiones de scores (B, H, T, T)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Corrección de estabilidad: Enmascarar de manera segura
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(), # GELU es más suave y ayuda a converger mejor que ReLU en caracteres
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout)
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Estructura Pre-LN Pura
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        # CORRECCIÓN: Eliminamos la multiplicación destructiva por d_model ** 0.5
        return self.positional_encoding(self.token_embedding(x))

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, dropout)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # CORRECCIÓN: Al usar Pre-LN, se necesita obligatoriamente una norma final antes de proyectar al vocabulario
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.output_layer = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x) # <-- Bloque de estabilidad final
        return self.output_layer(x)