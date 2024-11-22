import torch
from torch import nn


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
    def __init__(self, embed_size, mlp_size):
        super(Mlp, self).__init__()
        self.embed_size = embed_size
        self.mlp_size = mlp_size
        
        self.fc1 = nn.Linear(embed_size, mlp_size)
        self.fc2 = nn.Linear(mlp_size, embed_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
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
    