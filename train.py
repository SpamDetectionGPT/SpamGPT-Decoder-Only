import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import SpamGPTConfig, config_dict
from spamGPT import SpamGPT, generate
from inference import run_inference, enc, special_tokens
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
    def __init__(self, model, train_loader, val_loader, batch_size=32, epochs=10, lr=1e-4, gradient_accumulation_steps=2):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
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
        
        # Initialize accumulated metrics
        accumulated_loss = 0
        accumulated_token_acc = 0
        accumulated_seq_acc = 0
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = self.model(x)
            main_loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # --- Auxiliary loss for <SPAM>/<HAM> prediction after <SOP> ---
            sop_token_id = enc.encode("<SOP>", allowed_special={"<SOP>"})[0]
            special_tokens_ids = torch.tensor([special_tokens["<SPAM>"], special_tokens["<HAM>"]], device=x.device)
            sop_positions = (x == sop_token_id).nonzero(as_tuple=True)
            aux_loss = 0.0

            if len(sop_positions[0]) > 0:
                # Get logits and targets at <SOP> positions
                sop_logits = logits[sop_positions[0], sop_positions[1]]
                sop_targets = y[sop_positions[0], sop_positions[1]]
                # Mask for <SPAM>/<HAM> targets
                spam_ham_mask = torch.isin(sop_targets, special_tokens_ids)
                if spam_ham_mask.any():
                    aux_loss = self.criterion(sop_logits[spam_ham_mask], sop_targets[spam_ham_mask])
                    # You can adjust the weight (e.g., 2.0) as needed
                    loss = main_loss + 2.0 * aux_loss
                else:
                    loss = main_loss
            else:
                loss = main_loss
            # -------------------------------------------------------------

            # Normalize loss by gradient accumulation steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Calculate accuracies
            token_acc, sequence_acc = self.calculate_accuracy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1),
                x.view(-1)
            )
            
            # Accumulate metrics
            accumulated_loss += loss.item()
            accumulated_token_acc += token_acc
            accumulated_seq_acc += sequence_acc
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update metrics with the accumulated values
                total_loss += accumulated_loss * self.gradient_accumulation_steps
                total_token_accuracy += accumulated_token_acc / self.gradient_accumulation_steps
                total_sequence_accuracy += accumulated_seq_acc / self.gradient_accumulation_steps
                
                # Reset accumulated metrics
                accumulated_loss = 0
                accumulated_token_acc = 0
                accumulated_seq_acc = 0
            
            # Only update progress bar every 20 batches
            if batch_idx % 20 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}', 
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
            # Store the progress bar in a variable
            progress_bar =  tqdm(self.val_loader, desc="Validation")
            for batch_idx, (x, y) in enumerate(progress_bar):
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                token_acc, sequence_acc = self.calculate_accuracy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1),
                    x.view(-1)
                )
                
                total_loss += loss.item()
                total_token_accuracy += token_acc
                total_sequence_accuracy += sequence_acc
                
                # Now progress_bar is defined and can be used
                if batch_idx % 20 == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}', 
                        'token_acc': f'{token_acc:.4f}',
                        'seq_acc': f'{sequence_acc:.4f}'
                    })
                
        return (total_loss / len(self.val_loader), 
                total_token_accuracy / len(self.val_loader),
                total_sequence_accuracy / len(self.val_loader))

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            if (epoch + 1) % 10 == 0:
                print("\nRunning inference on test examples...")
                # Create a sample input for inference
                sample_input = torch.tensor([[enc.encode("[CLS] subject: estimated actuals for april 5")[0]]], device=device)
                run_inference(
                    input_tokens=sample_input,
                    max_length=100,
                    model=self.model,
                    temp=0.8,
                    enc=enc,
                    endtoken=enc.encode("<EOE>")[0]
                )
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
        batch_size=32 
    )


    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=32,
        epochs=50,
        lr=1e-4,
        gradient_accumulation_steps=4
    )
    
    trainer.train()