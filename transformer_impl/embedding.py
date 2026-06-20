import torch
from transformer_impl.position import get_position
from transformer_impl.config import ModelConfig

class TransformerEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, config: ModelConfig):
        super().__init__()
        emb_drop = config.embedding_dropout if config.embedding_dropout is not None else config.dropout
        self.token_embedding = torch.nn.Embedding(vocab_size, config.d_model)
        self.position_encoding = get_position(
            config.position.type,
            config.d_model,
            dropout=emb_drop,
            max_len=config.position.max_len,
        )

    def forward(self, x):
        return self.position_encoding(self.token_embedding(x))
