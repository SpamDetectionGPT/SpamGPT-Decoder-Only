# SpamGPT

A decoder-only transformer model for spam detection with comprehensive training and evaluation tooling.

## Overview

SpamGPT is a transformer-based model designed to detect spam messages. It uses a decoder-only architecture similar to GPT models, with special tokens to mark the start and end of messages, as well as spam/ham labels.

## Features

- Decoder-only transformer architecture
- Custom attention mechanism with causal masking
- Special token handling for message boundaries and labels
- Comprehensive training pipeline with validation
- GPU monitoring utilities
- Wandb integration for experiment tracking
- Configurable model parameters

## Model Architecture

- Block size: 256 tokens
- Vocabulary size: 50,263 tokens
- 6 transformer layers
- 6 attention heads
- 312 embedding dimensions
- LeakyReLU activation in MLP layers

## Training

The model is trained using a custom training loop that includes:
- Token-level and sequence-level accuracy metrics
- Validation after each epoch
- Checkpoint saving for best models
- Progress tracking with tqdm
- GPU memory monitoring

## Dataset Format

The model expects data in JSON format with the following structure:
```json
{
    "dataset": [
        {
            "text": "message content"
        }
    ]
}
```

Messages are processed with special tokens:
- `<SOE>`: Start of email
- `<EOE>`: End of email
- `<SOP>`: Start of prediction
- `<HAM>`: Ham label
- `<SPAM>`: Spam label
- `<EOP>`: End of prediction

## Usage

1. Prepare your data in the required JSON format
2. Configure model parameters in `config.py`
3. Run training:
```bash
python train.py
```

4. Monitor GPU usage:
```bash
python gpu_monitor.py
```

## Dependencies

- PyTorch
- tqdm
- wandb
- psutil

## License


