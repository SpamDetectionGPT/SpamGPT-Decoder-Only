import torch
import torch.nn as nn
from dataclasses import dataclass, asdict


@dataclass
class SpamGPTConfig:
    block_size: int = 256
    vocab_size: int = 50263
    n_layer: int = 12
    n_head: int = 6
    n_emb: int = 720
    
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

# Create a dictionary version of the config
config_dict = SpamGPTConfig().to_dict()

# Example usage:
if __name__ == "__main__":
    # Class version
    config = SpamGPTConfig()
    print("Class version:", config)
    
    # Dictionary version
    print("Dictionary version:", config_dict)
