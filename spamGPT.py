from config import SpamGPTConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken

base_encoding = tiktoken.get_encoding("gpt2")
special_tokens = {
    "<SOE>": 50257,  # Add one after the vocab size (GPT-2 vocab size is 50257)
    "<EOE>": 50258,
    "<SOP>": 50259,
    "<EOP>": 50260,
    "<SPAM>": 50261,
    "<HAM>": 50262,
}
enc = tiktoken.Encoding(
    name="gpt2-custom",
    pat_str=base_encoding._pat_str,
    mergeable_ranks=base_encoding._mergeable_ranks,
    special_tokens={**base_encoding._special_tokens, **special_tokens}
)


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


def run_inference(input_tokens, max_length, model, temp, enc, endtoken=None):
    model.eval()
    print("\n\n-----------------Inference-----------------\n")
    while input_tokens.size()[1] < max_length:
        if endtoken is not None and input_tokens[0, -1].item() == endtoken:
            break
        pred = model(input_tokens)
        logits = pred[:,-1, :]
        logits = logits/temp
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        print(enc.decode([next_token.squeeze().tolist()]), end="")
        input_tokens = torch.cat((input_tokens, next_token), dim=1)
    print("\n\n---------------End Inference---------------\n")
    return input_tokens


if __name__ == "__main__":
    #example use
    model = SpamGPT(SpamGPTConfig)
    x = torch.tensor([[50257, 4, 9]], dtype=torch.long)
    run_inference(x, 20, model, 0.7, enc)
    