import json
import torch
from config import SpamGPTConfig
from inference import enc
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random
from tqdm import tqdm
import numpy as np

class SpamDataset(Dataset):
    def __init__(self, ham_file: str, spam_file: str, block_size: int, encoder):
        self.block_size = block_size
        self.encoder = encoder
        
        # Define our special tokens
        self.special_tokens = ["<SOE>", "<EOE>", "<SOP>", "<HAM>", "<SPAM>", "<EOP>"]
        self.allowed_special = set(self.special_tokens)
        
        # Calculate the effective block size by subtracting the length of special tokens
        special_tokens_length = sum(len(self.encoder.encode(token, allowed_special=self.allowed_special)) 
                                  for token in self.special_tokens)
        self.effective_block_size = block_size - special_tokens_length
        
        # Load and combine datasets
        self.data = []
        
        # Load ham data
        with open(ham_file, 'r') as f:
            ham_data = json.load(f)
            
            for item in tqdm(ham_data['dataset'], desc="Loading ham data"):
                # Encode and truncate/pad the original text
                encodings = self.encoder.encode(item['text'], allowed_special=self.allowed_special)
                if len(encodings) > self.effective_block_size:
                    encodings = encodings[:self.effective_block_size]
                
                # Decode back to text and add special tokens
                text = self.encoder.decode(encodings)
                text = f"<SOE>{text}<EOE><SOP><HAM><EOP>"
                self.data.append(text)
        
        # Load spam data
        with open(spam_file, 'r') as f:
            spam_data = json.load(f)
            for item in tqdm(spam_data['dataset'], desc="Loading spam data"):
                # Encode and truncate/pad the original text
                encodings = self.encoder.encode(item['text'], allowed_special=self.allowed_special)
                if len(encodings) > self.effective_block_size:
                    encodings = encodings[:self.effective_block_size]
                
                # Decode back to text and add special tokens
                text = self.encoder.decode(encodings)
                text = f"<SOE>{text}<EOE><SOP><SPAM><EOP>"
                self.data.append(text)
        
        random.shuffle(self.data)
        
        # Combine all data into one long string
        self.combined_text = "".join(self.data)
        self.encoded_data = self.encoder.encode(self.combined_text, allowed_special=self.allowed_special)
        
        # Calculate total length and number of blocks
        self.total_length = len(self.encoded_data)
        self.num_blocks = self.total_length // self.block_size
        
        print(f"""
        General Information:
        Total tokens: {self.total_length}
        Number of blocks: {self.num_blocks}
        Block size: {self.block_size}
        Total ham examples: {len([text for text in self.data if "<HAM>" in text])}
        Total spam examples: {len([text for text in self.data if "<SPAM>" in text])}
        """)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        # Calculate start and end indices for this block
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        # Get the block of tokens
        x = torch.tensor(self.encoded_data[start_idx:end_idx], dtype=torch.long)
        
        # Create labels by shifting input by one position
        y = torch.cat([x[1:], torch.tensor([0])])
        
        return x, y

def create_dataloaders(ham_file: str, spam_file: str, block_size: int, encoder, 
                      batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation dataloaders for the spam dataset.
    
    Args:
        ham_file: Path to ham JSON file
        spam_file: Path to spam JSON file
        block_size: Maximum sequence length
        encoder: Tokenizer/encoder to use
        batch_size: Batch size for dataloaders
        train_split: Proportion of data to use for training
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset
    dataset = SpamDataset(ham_file, spam_file, block_size, encoder)
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(
        ham_file='combined_ham.json',
        spam_file='combined_spam.json',
        block_size=SpamGPTConfig.block_size,
        encoder=enc,
        batch_size=512)
    
    
    
