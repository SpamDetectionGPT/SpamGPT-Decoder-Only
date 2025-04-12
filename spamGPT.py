from config import SpamGPTConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_emb % config.n_head == 0

        self.c_attn = nn.Linear(config.n_emb, 3*config.n_emb)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_emb, dim=-1)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, config.n_emb*4)
        self.lrelu = nn.LeakyReLU()
        self.c_proj = nn.Linear(config.n_emb*4, config.n_emb)

    def forward(self, x):
        return self.c_proj(self.lrelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SpamGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_emb),
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)  # shape: (1, T)

        tok_emb = self.transformer['wte'](x)  # token embeddings
        pos_emb = self.transformer['wpe'](pos)  # position embeddings
        x = tok_emb + pos_emb

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        return self.lm_head(x)
    

if __name__ == "__main__":
    #example use
    model = SpamGPT(SpamGPTConfig)
    x = torch.tensor([[1, 4, 9]], dtype=torch.long)
    print(model(x))
    