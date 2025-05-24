import json
import torch
from config import SpamGPTConfig
from inference import enc
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional
import random
from tqdm import tqdm
import numpy as np
import os

class SpamDataset(Dataset):
    def __init__(self, ham_file: str, spam_file: str, block_size: int, encoder):
        self.block_size = block_size
        self.encoder = encoder
        random.seed(42)
        # Define our special tokens
        self.special_tokens = ["<SOE>", "<EOE>", "<SOP>", "<HAM>", "<SPAM>", "<EOP>"]
        self.allowed_special = set(self.special_tokens)
        print(self.encoder.encode("<SOE><EOE><SOP><HAM><EOP>", allowed_special=self.allowed_special))
        
        # Calculate the effective block size by subtracting the length of special tokens
        special_tokens_length = sum(len(self.encoder.encode(token, allowed_special=self.allowed_special)) 
                                  for token in self.special_tokens)
        self.effective_block_size = block_size - special_tokens_length
        
        # Load and combine datasets
        self.data = []
        
        # Only show progress bars on master process (rank 0) or non-DDP runs
        ddp_rank = int(os.environ.get('RANK', 0))
        show_progress = ddp_rank == 0
        
        # Load ham data
        with open(ham_file, 'r') as f:
            ham_data = json.load(f)
            
            for item in tqdm(ham_data['dataset'], desc="Loading ham data", disable=not show_progress):
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
            for item in tqdm(spam_data['dataset'], desc="Loading spam data", disable=not show_progress):
                # Encode and truncate/pad the original text
                encodings = self.encoder.encode(item['text'], allowed_special=self.allowed_special)
                if len(encodings) > self.effective_block_size:
                    encodings = encodings[:self.effective_block_size]
                
                # Decode back to text and add special tokens
                text = self.encoder.decode(encodings)
                text = f"<SOE>{text}<EOE><SOP><SPAM><EOP>"
                self.data.append(text)
        
        if show_progress:
            print(self.data[0])
        
        random.shuffle(self.data)
        
        # Combine all data into one long string
        self.combined_text = "".join(self.data)
        self.encoded_data = self.encoder.encode(self.combined_text, allowed_special=self.allowed_special)
        
        # Calculate total length and number of blocks
        self.total_length = len(self.encoded_data)
        self.num_blocks = self.total_length // self.block_size
        
        if show_progress:
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

def create_dataloaders(train_ham_file: str, train_spam_file: str, 
                      test_ham_file: str, test_spam_file: str,
                      block_size: int, encoder, 
                      batch_size: int = 32, seed: int = 42,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """
    Creates train and validation dataloaders for the spam dataset with DDP support.
    
    Args:
        train_ham_file: Path to training ham JSON file
        train_spam_file: Path to training spam JSON file
        test_ham_file: Path to test ham JSON file
        test_spam_file: Path to test spam JSON file
        block_size: Maximum sequence length
        encoder: Tokenizer/encoder to use
        batch_size: Batch size for dataloaders (per GPU in DDP mode)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, train_sampler, val_sampler)
        Samplers will be None if not using DDP
    """
    # Create train dataset
    train_dataset = SpamDataset(train_ham_file, train_spam_file, block_size, encoder)
    
    # Create test dataset
    test_dataset = SpamDataset(test_ham_file, test_spam_file, block_size, encoder)
    
    # Check if we're in DDP mode
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=seed
        )
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,  # Usually don't shuffle test data
            seed=seed
        )
        
        # Create dataloaders with samplers
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        # Non-DDP mode: use regular DataLoader with shuffle
        train_sampler = None
        test_sampler = None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False
        )
    
    return train_loader, test_loader, train_sampler, test_sampler


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader, train_sampler, test_sampler = create_dataloaders(
        train_ham_file='train_data_ham.json',
        train_spam_file='train_data_spam.json',
        test_ham_file='test_data_ham.json',
        test_spam_file='test_data_spam.json',
        block_size=SpamGPTConfig.block_size,
        encoder=enc,
        batch_size=512,
        num_workers=4
    )
    
    print(f"Train loader length: {len(train_loader)}")
    print(f"Test loader length: {len(test_loader)}")
    print(f"Using distributed samplers: {train_sampler is not None}")