import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import SpamGPTConfig
from inference import run_inference, enc

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.latest_attn = None

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=-1)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # CRITICAL FIX: Apply causal mask always, not just during training
        att = att.masked_fill(
            self.bias[:, :, :T, :T] == 0, float('-inf')
        )
        
        att = F.softmax(att, dim=-1)
        self.latest_attn = att.detach()  
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = SpamGPTConfig(**config_args)
        model = SpamGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        embedding_dim = model.transformer.wte.weight.shape[1]

        # Create new larger embeddings
        new_wte = nn.Embedding(50263, embedding_dim)
        new_lm_head = nn.Linear(embedding_dim, 50263, bias=False)

        # Copy old weights
        with torch.no_grad():
            new_wte.weight[:50257] = model.transformer.wte.weight
            new_lm_head.weight[:50257] = model.lm_head.weight
            
            # Initialize new token embeddings
            nn.init.normal_(new_wte.weight[50257:], mean=0.0, std=0.02)
            nn.init.normal_(new_lm_head.weight[50257:], mean=0.0, std=0.02)

        # Replace layers
        model.transformer.wte = new_wte
        model.lm_head = new_lm_head
        model.config.vocab_size = 50263
        return model

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
):
    """
    Generates new tokens from the model given a context idx.

    Args:
        model: Instance of SpamGPT or compatible transformer model.
        idx (torch.LongTensor): Input token IDs of shape (B, T).
        max_new_tokens (int): Number of tokens to generate.
        temperature (float): Sampling temperature; higher means more random.
        top_k (int, optional): If specified, use top-k sampling.

    Returns:
        torch.LongTensor: Generated token IDs of shape (B, T+max_new_tokens).
    """
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        # ensure context length does not exceed the model's block size
        context = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward pass
        logits = model(context)
        # focus on last time-step logits
        logits = logits[:, -1, :] / temperature

        # optionally apply top-k filtering
        if top_k is not None:
            # find the top-k logits
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            # mask logits not in top-k
            min_topk = topk_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_topk, torch.full_like(logits, -float('Inf')), logits)

        # convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample or take the most likely token
        next_token = torch.multinomial(probs, num_samples=1)

        # append to sequence
        idx = torch.cat((idx, next_token), dim=1)
    return idx




if __name__ == "__main__":
    config = SpamGPTConfig()
    model = SpamGPT.from_pretrained("gpt2")
    print(model)
    text = run_inference(torch.tensor([enc.encode("Hello I'm ")]), 100, model, 1, enc, "<EOP>")
    print(text.shape)
    text = enc.decode(text.cpu().numpy().tolist()[0])
    print(text)
    print(len(text))
    
