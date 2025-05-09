# SpamGPT

A decoder-only transformer model for spam detection, built with PyTorch. This implementation uses a GPT-style architecture to classify and generate email content.

## Features

- Decoder-only transformer architecture with configurable parameters
- Custom tokenizer with special tokens for email classification
- Dataset handling for both spam and ham (legitimate) emails
- Training pipeline with validation support
- Inference capabilities with temperature-controlled generation

## Model Architecture

- Block size: 256 tokens
- Vocabulary size: 50,263 tokens (including custom special tokens)
- 6 transformer layers
- 6 attention heads
- 312 embedding dimensions

## Special Tokens

The model uses the following special tokens:
- `<SOE>`: Start of Email
- `<EOE>`: End of Email
- `<SOP>`: Start of Prediction
- `<EOP>`: End of Prediction
- `<SPAM>`: Spam classification token
- `<HAM>`: Ham (legitimate) classification token

## Dataset Structure

The model expects two JSON files:
- `combined_ham.json`: Contains legitimate emails
- `combined_spam.json`: Contains spam emails

Each file should follow the format:
```json
{
    "dataset": [
        {"text": "email content..."},
        ...
    ]
}
```

### Data Sources
The model can be trained on various email datasets:
- Enron Dataset
- TREC 2007 Spam Corpus

Note: Due to the large size of these datasets, they are not included in the repository. You'll need to download them separately and place them in the appropriate directories.

## Setup and Usage

1. Install dependencies:
```bash
pip install torch tiktoken tqdm numpy pandas
```

2. Download and prepare your dataset:
   - Download the Enron and/or TREC 2007 datasets
   - Place them in the `datasets/` directory
   - Run the data preprocessing scripts to generate the combined JSON files

3. Training:
```python
from dataset import create_dataloaders
from train import Trainer
from spamGPT import SpamGPT
from config import SpamGPTConfig

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    ham_file='combined_ham.json',
    spam_file='combined_spam.json',
    block_size=SpamGPTConfig.block_size,
    encoder=enc,
    batch_size=32
)

# Initialize and train model
model = SpamGPT(SpamGPTConfig())
trainer = Trainer(model, train_loader, val_loader, batch_size=32, epochs=10, lr=1e-4)
```

4. Inference:
```python
from inference import run_inference

# Run inference with temperature control
output = run_inference(input_tokens, max_length, model, temp=0.7, enc=enc)
```

## Project Structure

- `spamGPT.py`: Core model implementation
- `config.py`: Model configuration
- `dataset.py`: Dataset handling and preprocessing
- `train.py`: Training pipeline
- `inference.py`: Inference utilities
- `synth_dataset_ham.py`: Dataset synthesis tool

## Data Management

This project uses large datasets that exceed GitHub's file size limits. To handle this:

1. The raw datasets are not included in the repository
2. You'll need to download the datasets separately
3. The combined JSON files (`combined_ham.json` and `combined_spam.json`) are generated during preprocessing
4. These files should be added to `.gitignore` to prevent accidental commits

To download the datasets:
1. Enron Dataset: [Add download link]
2. TREC 2007 Spam Corpus: [Add download link]


