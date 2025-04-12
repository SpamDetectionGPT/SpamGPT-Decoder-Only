import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SpamGPTConfig:
    block_size = 256
    vocab_size = 50262
    n_layer = 12
    n_head = 12
    n_emb = 768


