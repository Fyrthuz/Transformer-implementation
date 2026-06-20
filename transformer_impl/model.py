import torch
from transformer_impl.config import ExperimentConfig
from transformer_impl.embedding import TransformerEmbedding
from transformer_impl.blocks import TransformerBlock

class Transformer(torch.nn.Module):
    def __init__(self, config: ExperimentConfig, vocab_size: int):
        super().__init__()
        self.cfg = config
        self.embedding = TransformerEmbedding(vocab_size, config.model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(config.model, layer_idx=i)
            for i in range(config.model.num_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(config.model.d_model)
        self.output_layer = torch.nn.Linear(config.model.d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.output_layer(x)

    def auxiliary_losses(self):
        losses = []
        for layer in self.layers:
            losses.extend(layer.auxiliary_losses())
        return losses

    def generate_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
