import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import SpamGPTConfig, config_dict
from spamGPT import SpamGPT, generate
from inference import run_inference, enc
import zipfile
import os
import pandas as pd
import json
from dataset import create_dataloaders
from tqdm import tqdm
import time
import wandb

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print(device)

run = wandb.init(project="spamGPT", config=config_dict)


class Trainer:
    def __init__(self, model, train_loader, val_loader, batch_size=32, epochs=10, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def calculate_accuracy(self, logits, targets, input_ids):
        # Regular token-level accuracy
        predictions = torch.argmax(logits, dim=-1)
        token_accuracy = (predictions == targets).float().mean()
        
        # Find positions of <SOP> tokens in the input sequences
        sop_token_id = enc.encode("<SOP>", allowed_special={"<SOP>"})[0]  # Get the token ID for <SOP>
        sop_positions = (input_ids == sop_token_id).nonzero(as_tuple=True)
        
        # Get predictions and targets at <SOP> positions
        sop_predictions = predictions[sop_positions]
        sop_targets = targets[sop_positions]
        
        # Calculate sequence-level accuracy only for positions with <SOP>
        if len(sop_predictions) > 0:
            sequence_accuracy = (sop_predictions == sop_targets).float().mean()
        else:
            sequence_accuracy = torch.tensor(0.0)
        
        return token_accuracy.item(), sequence_accuracy.item()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_token_accuracy = 0
        total_sequence_accuracy = 0
        total_sequences = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = self.model(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Calculate accuracies
            token_acc, sequence_acc = self.calculate_accuracy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1),
                x.view(-1)  # Pass the input tokens to find <SOP> positions
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_token_accuracy += token_acc
            total_sequence_accuracy += sequence_acc
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'token_acc': f'{token_acc:.4f}',
                'seq_acc': f'{sequence_acc:.4f}'
            })
            
        return (total_loss / len(self.train_loader), 
                total_token_accuracy / len(self.train_loader),
                total_sequence_accuracy / len(self.train_loader))

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_token_accuracy = 0
        total_sequence_accuracy = 0
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                token_acc, sequence_acc = self.calculate_accuracy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1),
                    x.view(-1)  # Pass the input tokens to find <SOP> positions
                )
                
                total_loss += loss.item()
                total_token_accuracy += token_acc
                total_sequence_accuracy += sequence_acc
                
        return (total_loss / len(self.val_loader), 
                total_token_accuracy / len(self.val_loader),
                total_sequence_accuracy / len(self.val_loader))

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_token_acc, train_seq_acc = self.train_epoch()
            
            # Validation
            val_loss, val_token_acc, val_seq_acc = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Log metrics to wandb
            wandb.log({
                'train_loss': train_loss,
                'train_token_accuracy': train_token_acc,
                'train_sequence_accuracy': train_seq_acc,
                'val_loss': val_loss,
                'val_token_accuracy': val_token_acc,
                'val_sequence_accuracy': val_seq_acc,
                'epoch_time': epoch_time
            })
            
            # Print epoch statistics
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Token Acc: {train_token_acc:.4f} | Train Seq Acc: {train_seq_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Token Acc: {val_token_acc:.4f} | Val Seq Acc: {val_seq_acc:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_token_accuracy': train_token_acc,
                    'train_sequence_accuracy': train_seq_acc,
                    'val_loss': val_loss,
                    'val_token_accuracy': val_token_acc,
                    'val_sequence_accuracy': val_seq_acc,
                }
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
                print(f"Saved new best model with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Initialize model
    model = SpamGPT(SpamGPTConfig()).to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        ham_file='combined_ham.json',
        spam_file='combined_spam.json',
        block_size=SpamGPTConfig.block_size,
        encoder=enc,
        batch_size=128
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=128,
        epochs=50,
        lr=1e-4
    )
    
    # Start training
    trainer.train()