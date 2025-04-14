import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SpamGPTConfig:
    block_size = 256
    vocab_size = 50263
    n_layer = 6
    n_head = 6
    n_emb = 312


