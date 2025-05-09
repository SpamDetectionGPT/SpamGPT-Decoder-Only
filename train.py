import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import SpamGPTConfig
from spamGPT import SpamGPT, generate
from inference import run_inference, enc
import zipfile
import os
import pandas as pd
import json
from dataset import create_dataloaders


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = SpamGPT(SpamGPTConfig()).to(device)

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=32, epochs=10, lr=1e-4):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = batch



# Replace the Trainer initialization with:
train_loader, val_loader = create_dataloaders(
    ham_file='combined_ham.json',
    spam_file='combined_spam.json',
    block_size=SpamGPTConfig.block_size,
    encoder=enc,
    batch_size=32
)

trainer = Trainer(model, train_loader, val_loader, batch_size=32, epochs=10, lr=1e-4)
