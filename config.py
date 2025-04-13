import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SpamGPTConfig:
    latent_dim_kv = 512
    block_size = 256
    vocab_size = 50263
    n_layer = 12
    n_head = 12
    n_emb = 768


