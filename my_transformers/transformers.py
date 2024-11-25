import torch as th
import torch.nn as nn
from .nn import Block
import torch.nn.functional as F
class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_size),
            wpe = nn.Embedding(config.max_len, config.embed_size),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embed_size)
            )
        )
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)

        # sharing weights between token embedding and output layer
        self.self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx, target=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Input tokens length {T} exceeds maximum length {self.config.block_size}"
        
        # Embedding of tokens and positional encoding
        pos = th.arange(T, device=idx.device)
        tok_embed = self.transformer['wte'](idx) # tokens embeddings of B, T, embed_size
        pos_embed = self.transformer['wpe'](pos) # positions embeddings of T, embed_size
        x = tok_embed + pos_embed # B, T, embed_size
        
        # feed forward through transformer blocks
        for block in self.transformer['blocks']:
            x = block(x) # B, T, embed_size
            
        # Layer normalization
        x = self.transformer['ln_f'](x) # B, T, embed_size
        
        # Output logits for next token prediction from lm_head
        logits = self.lm_head(x) # B, T, vocab_size
        
        # Compute loss if target is provided
        loss = None
        if target is not None:
            tmp = logits.view(-1, logits.size(-1)) # B*T, vocab_size
            loss = F.cross_entropy(tmp, target.view(-1)) # scalar loss for B*T predictions
        return logits, loss

    def save_model(self, path):
        th.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(th.load(path))
            
        