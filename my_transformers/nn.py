import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head_size = head_size
        
        self.keys = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        
        self.fc_out = nn.Linear(self.head_size, embed_size)
        self.register_buffer('tril', torch.tril(torch.ones((x.shape[1], x.shape[1]))))
        
    def forward(self, x):
        k = self.keys(x) # B, block_size, head_size
        q = self.queries(x) # B, block_size, head_size
        v = self.value(x)


        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, block_size, head_size) @ (B, head_size, block_size) -> (B, block_size, block_size)
        wei = wei.masked_fill(self.tril == 0, float('-inf')) # B, block_size, block_size
        wei = torch.softmax(wei, dim=-1)
        out = wei @ v
        out = self.fc_out(out)

    
        return out

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.embed_size = config.embed_size
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size * 4),
            nn.GELU(),
            nn.Linear(self.embed_size * 4, self.embed_size)
        )
        
    def forward(self, x):
        out = self.mlp(x)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, head_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.head_size = head_size
        self.num_heads = num_heads
        
        self.attentions = nn.ModuleList([SelfAttention(embed_size, head_size) for _ in range(num_heads)])
        self.fc_out = nn.Linear(num_heads * head_size, embed_size)
        
    def forward(self, x):
        out = torch.cat([attn(x) for attn in self.attentions], dim=-1)
        out = self.fc_out(out)
        return out
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()
        assert config.embed_size % config.num_heads == 0
        
        # self-attention values (key, query, value)
        self.c_atten = nn.Linear(config.embed_size, 3*config.head_size*config.num_heads) # head_size * num_heads = embed_size
        
        # output projection
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        
        self.embed_size = config.embed_size
        self.num_heads = config.num_heads
        
        self.register_buffer('bias',
                             torch.tril(torch.ones((config.block_size, config.block_size))).view(1, 1, config.block_size, config.block_size)
                             )
        
    def forward(self, x):
        # self-attention
        B, T, C = x.shape # Batch, sequence length, embed size
        kqv = self.c_atten(x)
        k, q, v = kqv.split(self.embed_size, dim=2) # each in shape B, T, head_size*num_heads 
        
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2) # B, num_heads, T, head_size
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2) # B, num_heads, T, head_size
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2) # B, num_heads, T, head_size

        # old way of calculating attention
        # wei = q @ k.transpose(-2, -1) * self.embed_size**-0.5
        # wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # B, num_heads, T, T
        # wei = torch.softmax(wei, dim=-1)
        # out = wei @ v # B, num_heads, T, head_size
        
        # new way of calculating attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C) # B, T, C (concatenated head_size*num_heads)
        out = self.c_proj(out)
        return out

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.ln2 = nn.LayerNorm(config.embed_size)
        self.mlp = Mlp(config)
        self.self_attn = SelfAttentionBlock(config)
        
    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    