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
import aiohttp
import asyncio
import os
from dotenv import load_dotenv
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

load_dotenv()

async def send_discord_webhook(text):
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("No Discord webhook URL found.")
        return
    
    async with aiohttp.ClientSession() as session:
        await session.post(webhook_url, json={"content": f"Model Inference:\n```\n{text}\n```"})

# DDP setup
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# Only initialize wandb on master process
if master_process:
    run = wandb.init(project="spamGPT", config=config_dict)

# Set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device_type = "cuda" if device.startswith("cuda") else "cpu"

class Trainer:
    def __init__(self, model, train_loader, val_loader, train_sampler=None, val_sampler=None, 
                 batch_size=32, epochs=10, lr=1e-4, gradient_accumulation_steps=2):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = "checkpoints"
        if master_process:
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
        total_aux_loss = 0
        total_classification_accuracy = 0
        num_batches = 0
        num_classification_batches = 0  # Track batches with classification
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(device), y.to(device)
            
            # Set require_backward_grad_sync for DDP
            if ddp:
                self.model.require_backward_grad_sync = ((batch_idx + 1) % self.gradient_accumulation_steps == 0)
            
            # Forward pass
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = self.model(x)
                main_loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Auxiliary loss for <HAM>/<SPAM> prediction after <SOP>
            sop_token_id = enc.encode("<SOP>", allowed_special={"<SOP>"})[0]
            ham_token_id = special_tokens["<HAM>"]
            spam_token_id = special_tokens["<SPAM>"]
            
            # Find all positions where <SOP> appears in the input
            batch_size, seq_len = x.shape
            sop_mask = (x == sop_token_id)
            
            # Create a mask for valid positions (not the last token in sequence)
            valid_sop_mask = sop_mask[:, :-1]  # Exclude last position
            
            # Get positions where we have <SOP> and can predict next token
            sop_positions = valid_sop_mask.nonzero(as_tuple=False)
            
            if len(sop_positions) > 0:
                # Get batch and sequence indices for positions after <SOP>
                batch_indices = sop_positions[:, 0]
                seq_indices = sop_positions[:, 1] + 1  # +1 to get position after <SOP>
                
                # Extract logits at positions after <SOP>
                classification_logits = logits[batch_indices, seq_indices]
                
                # Get the actual targets at these positions
                classification_targets = y[batch_indices, seq_indices]
                
                # Create a mask for valid classification targets (should be <HAM> or <SPAM>)
                valid_targets_mask = (classification_targets == ham_token_id) | (classification_targets == spam_token_id)
                
                if valid_targets_mask.any():
                    # Filter to only valid classification positions
                    valid_classification_logits = classification_logits[valid_targets_mask]
                    valid_classification_targets = classification_targets[valid_targets_mask]
                    
                    # Calculate auxiliary loss only for <HAM>/<SPAM> predictions
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        aux_loss = self.criterion(valid_classification_logits, valid_classification_targets)
                    
                    # Calculate classification accuracy
                    predictions = torch.argmax(valid_classification_logits, dim=-1)
                    classification_acc = (predictions == valid_classification_targets).float().mean()
                    
                    # Weight the auxiliary loss (you can adjust this weight)
                    aux_loss_weight = 2.0  # Increase importance of classification
                    weighted_aux_loss = aux_loss * aux_loss_weight
                    
                    total_aux_loss += aux_loss.item()
                    total_classification_accuracy += classification_acc.item()
                    num_classification_batches += 1
                else:
                    weighted_aux_loss = 0.0
                    classification_acc = 0.0
            else:
                weighted_aux_loss = 0.0
                classification_acc = 0.0
            
            # Combine main loss and auxiliary loss
            total_loss_value = main_loss + weighted_aux_loss
            
            # Normalize loss by gradient accumulation steps
            loss = total_loss_value / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Calculate accuracies
            token_acc, sequence_acc = self.calculate_accuracy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1),
                x.view(-1)
            )
            
            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_token_accuracy += token_acc
            total_sequence_accuracy += sequence_acc
            num_batches += 1
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping for stability with larger effective batch size
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Print progress only on master process
                if master_process:
                    print(f'Batch {batch_idx + 1}: loss={loss.item() * self.gradient_accumulation_steps:.4f}, '
                        f'token_acc={token_acc:.4f}, seq_acc={sequence_acc:.4f}, '
                        f'class_acc={classification_acc if isinstance(classification_acc, float) else classification_acc.item():.4f}')
        
        # Aggregate metrics across all processes
        if ddp:
            # Convert to tensors for all_reduce
            total_loss_tensor = torch.tensor(total_loss, device=device)
            total_token_accuracy_tensor = torch.tensor(total_token_accuracy, device=device)
            total_sequence_accuracy_tensor = torch.tensor(total_sequence_accuracy, device=device)
            total_aux_loss_tensor = torch.tensor(total_aux_loss, device=device)
            total_classification_accuracy_tensor = torch.tensor(total_classification_accuracy, device=device)
            num_batches_tensor = torch.tensor(num_batches, dtype=torch.long, device=device)
            num_classification_batches_tensor = torch.tensor(num_classification_batches, dtype=torch.long, device=device)
            
            # All-reduce
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_token_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_sequence_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_aux_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_classification_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_classification_batches_tensor, op=dist.ReduceOp.SUM)
            
            # Convert back to scalars
            total_loss = total_loss_tensor.item()
            total_token_accuracy = total_token_accuracy_tensor.item()
            total_sequence_accuracy = total_sequence_accuracy_tensor.item()
            total_aux_loss = total_aux_loss_tensor.item()
            total_classification_accuracy = total_classification_accuracy_tensor.item()
            num_batches = num_batches_tensor.item()
            num_classification_batches = num_classification_batches_tensor.item()
        
        # Calculate averages over all batches
        avg_loss = total_loss / num_batches
        avg_token_accuracy = total_token_accuracy / num_batches
        avg_sequence_accuracy = total_sequence_accuracy / num_batches
        avg_aux_loss = total_aux_loss / num_classification_batches if num_classification_batches > 0 else 0
        avg_classification_accuracy = total_classification_accuracy / num_classification_batches if num_classification_batches > 0 else 0
        
        return avg_loss, avg_token_accuracy, avg_sequence_accuracy, avg_aux_loss, avg_classification_accuracy


    def validate(self):
        self.model.eval()
        total_loss = 0
        total_token_accuracy = 0
        total_sequence_accuracy = 0
        total_aux_loss = 0
        total_classification_accuracy = 0
        num_batches = 0
        num_classification_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x, y = x.to(device), y.to(device)
                
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = self.model(x)
                    main_loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # Calculate auxiliary loss for validation too
                sop_token_id = enc.encode("<SOP>", allowed_special={"<SOP>"})[0]
                ham_token_id = special_tokens["<HAM>"]
                spam_token_id = special_tokens["<SPAM>"]
                
                batch_size, seq_len = x.shape
                sop_mask = (x == sop_token_id)
                valid_sop_mask = sop_mask[:, :-1]
                sop_positions = valid_sop_mask.nonzero(as_tuple=False)
                
                if len(sop_positions) > 0:
                    batch_indices = sop_positions[:, 0]
                    seq_indices = sop_positions[:, 1] + 1
                    
                    classification_logits = logits[batch_indices, seq_indices]
                    classification_targets = y[batch_indices, seq_indices]
                    
                    valid_targets_mask = (classification_targets == ham_token_id) | (classification_targets == spam_token_id)
                    
                    if valid_targets_mask.any():
                        valid_classification_logits = classification_logits[valid_targets_mask]
                        valid_classification_targets = classification_targets[valid_targets_mask]
                        
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            aux_loss = self.criterion(valid_classification_logits, valid_classification_targets)
                        predictions = torch.argmax(valid_classification_logits, dim=-1)
                        classification_acc = (predictions == valid_classification_targets).float().mean()
                        
                        total_aux_loss += aux_loss.item()
                        total_classification_accuracy += classification_acc.item()
                        num_classification_batches += 1
                
                token_acc, sequence_acc = self.calculate_accuracy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1),
                    x.view(-1)
                )
                
                total_loss += main_loss.item()
                total_token_accuracy += token_acc
                total_sequence_accuracy += sequence_acc
                num_batches += 1
                
                if master_process and (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    print(f'Validation Batch {batch_idx + 1}: loss={main_loss.item():.4f}, '
                        f'token_acc={token_acc:.4f}, seq_acc={sequence_acc:.4f}')
        
        # Aggregate metrics across all processes
        if ddp:
            # Convert to tensors for all_reduce
            total_loss_tensor = torch.tensor(total_loss, device=device)
            total_token_accuracy_tensor = torch.tensor(total_token_accuracy, device=device)
            total_sequence_accuracy_tensor = torch.tensor(total_sequence_accuracy, device=device)
            total_aux_loss_tensor = torch.tensor(total_aux_loss, device=device)
            total_classification_accuracy_tensor = torch.tensor(total_classification_accuracy, device=device)
            num_batches_tensor = torch.tensor(num_batches, dtype=torch.long, device=device)
            num_classification_batches_tensor = torch.tensor(num_classification_batches, dtype=torch.long, device=device)
            
            # All-reduce
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_token_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_sequence_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_aux_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_classification_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_classification_batches_tensor, op=dist.ReduceOp.SUM)
            
            # Convert back to scalars
            total_loss = total_loss_tensor.item()
            total_token_accuracy = total_token_accuracy_tensor.item()
            total_sequence_accuracy = total_sequence_accuracy_tensor.item()
            total_aux_loss = total_aux_loss_tensor.item()
            total_classification_accuracy = total_classification_accuracy_tensor.item()
            num_batches = num_batches_tensor.item()
            num_classification_batches = num_classification_batches_tensor.item()
        
        avg_loss = total_loss / num_batches
        avg_token_accuracy = total_token_accuracy / num_batches
        avg_sequence_accuracy = total_sequence_accuracy / num_batches
        avg_aux_loss = total_aux_loss / num_classification_batches if num_classification_batches > 0 else 0
        avg_classification_accuracy = total_classification_accuracy / num_classification_batches if num_classification_batches > 0 else 0
        
        return avg_loss, avg_token_accuracy, avg_sequence_accuracy, avg_aux_loss, avg_classification_accuracy


    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Set epoch for distributed samplers to ensure proper shuffling
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(epoch)
                
            start_time = time.time()
            if master_process and (((epoch + 1) % 10 == 0) or epoch == 0):
                print("\nRunning inference on test examples...")
                # Create a sample input for inference
                # Use the base model (not DDP) for inference
                inference_model = self.model.module if hasattr(self.model, 'module') else self.model
                sample_input = torch.tensor([[enc.encode("[CLS] subject: estimated actuals for april 5")[0]]], device=device)
                result = run_inference(
                    input_tokens=sample_input,
                    max_length=100,
                    model=inference_model,
                    temp=0.8,
                    enc=enc,
                    endtoken=enc.encode("<EOE>", allowed_special={"<EOE>"})[0]
                )
                result = enc.decode(result.squeeze().tolist())
                asyncio.run(send_discord_webhook(result))
                
            # Training
            train_loss, train_token_acc, train_seq_acc, train_aux_loss, train_class_acc = self.train_epoch()
            
            # Validation
            val_loss, val_token_acc, val_seq_acc, val_aux_loss, val_class_acc = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Log all metrics to wandb at the end of each epoch (only on master)
            if master_process:
                wandb.log({
                    # Training metrics
                    'train_loss': train_loss,
                    'train_accuracy': train_token_acc,  # token-level accuracy
                    'train_sequence_accuracy': train_seq_acc,
                    'train_aux_loss': train_aux_loss,
                    'train_classification_accuracy': train_class_acc,
                    
                    # Validation/Test metrics
                    'test_loss': val_loss,
                    'test_accuracy': val_token_acc,  # token-level accuracy
                    'test_sequence_accuracy': val_seq_acc,
                    'test_aux_loss': val_aux_loss,
                    'test_classification_accuracy': val_class_acc,
                    
                    # Other metrics
                    'epoch': epoch,
                    'epoch_time': epoch_time,
                    'effective_batch_size': self.batch_size * ddp_world_size,  # Show effective batch size
                })
                
                # Print epoch statistics
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.epochs}")
                print(f"{'-'*60}")
                print(f"Train - Loss: {train_loss:.4f} | Token Acc: {train_token_acc:.4f} | Seq Acc: {train_seq_acc:.4f}")
                print(f"      - Aux Loss: {train_aux_loss:.4f} | Class Acc: {train_class_acc:.4f}")
                print(f"Test  - Loss: {val_loss:.4f} | Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f}")
                print(f"      - Aux Loss: {val_aux_loss:.4f} | Class Acc: {val_class_acc:.4f}")
                print(f"Time: {epoch_time:.2f}s | Effective Batch Size: {self.batch_size * ddp_world_size}")
                print(f"{'='*60}\n")
            
            if master_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the underlying model (not the DDP wrapper)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_token_accuracy': train_token_acc,
                    'train_sequence_accuracy': train_seq_acc,
                    'train_aux_loss': train_aux_loss,
                    'train_classification_accuracy': train_class_acc,
                    'val_loss': val_loss,
                    'val_token_accuracy': val_token_acc,
                    'val_sequence_accuracy': val_seq_acc,
                    'val_aux_loss': val_aux_loss,
                    'val_classification_accuracy': val_class_acc,
                }
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
                print(f"✓ Saved new best model with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Initialize model
    model = SpamGPT.from_pretrained("gpt2")
    model = model.to(device)
    
    # Wrap model in DDP if using distributed training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model
    
    # Batch size settings
    total_batch_size = 32  # Total batch size across all GPUs
    assert total_batch_size % ddp_world_size == 0, "make sure total_batch_size is divisible by ddp_world_size"
    batch_size_per_gpu = total_batch_size // ddp_world_size
    
    if master_process:
        print(f"Total batch size: {total_batch_size}")
        print(f"Batch size per GPU: {batch_size_per_gpu}")
        print(f"Number of GPUs: {ddp_world_size}")
    
    # Create dataloaders with the per-GPU batch size and distributed sampling
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        train_ham_file='train_data_ham.json',
        train_spam_file='train_data_spam.json',
        test_ham_file='test_data_ham.json',
        test_spam_file='test_data_spam.json',
        block_size=SpamGPTConfig.block_size,
        encoder=enc,
        batch_size=batch_size_per_gpu,  # Use per-GPU batch size
        num_workers=4  # Adjust based on your system
    )

    # Calculate gradient accumulation steps to achieve desired total batch size
    # if you want an even larger effective batch size
    gradient_accumulation_steps = 8  # Adjust as needed
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        batch_size=batch_size_per_gpu,
        epochs=15,
        lr=1e-4,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    trainer.train()
    
    if ddp:
        destroy_process_group()