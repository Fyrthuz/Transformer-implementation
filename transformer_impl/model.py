import torch
from transformer_impl.config import ExperimentConfig
from transformer_impl.embedding import TransformerEmbedding
from transformer_impl.blocks import TransformerBlock

class Transformer(torch.nn.Module):
    def __init__(self, config: ExperimentConfig, vocab_size: int):
        super().__init__()
        self.cfg = config
        self.vocab_size = vocab_size
        self.embedding = TransformerEmbedding(vocab_size, config.model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(config.model, layer_idx=i, num_layers=config.model.num_layers)
            for i in range(config.model.num_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(config.model.d_model)
        self.output_layer = torch.nn.Linear(config.model.d_model, vocab_size)
        self.v_head = None

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.output_layer(x)

    def forward_hidden(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln_f(x)

    def forward_value(self, x, mask=None):
        hidden = self.forward_hidden(x, mask)
        if self.v_head is None:
            self.add_value_head()
        return self.v_head(hidden).squeeze(-1)

    def add_value_head(self):
        if self.v_head is None:
            self.v_head = torch.nn.Linear(self.cfg.model.d_model, 1)
            if hasattr(self, 'output_layer') and self.output_layer.weight.device:
                self.v_head = self.v_head.to(self.output_layer.weight.device)

    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, eos_token_id=None, device='cpu'):
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.size(1) > 2048:
                    input_cond = generated[:, -2048:]
                else:
                    input_cond = generated
                mask = self.generate_causal_mask(input_cond.size(1), device)
                logits = self(input_cond, mask=mask)
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_ids], dim=1)
                if eos_token_id is not None and (next_ids == eos_token_id).any():
                    break
        return generated

    def auxiliary_losses(self):
        losses = []
        for layer in self.layers:
            losses.extend(layer.auxiliary_losses())
        return losses

    def generate_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
