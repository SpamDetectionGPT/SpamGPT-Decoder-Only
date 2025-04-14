import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import SpamGPTConfig
from spamGPT import SpamGPT
from inference import run_inference, enc
import zipfile
import os
import pandas as pd


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = SpamGPT(SpamGPTConfig()).to(device)

with zipfile.ZipFile("./spam-mails-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("./")

df = pd.read_csv("./spam_ham_dataset.csv")
labels = list(df["label"])
text = list(df["text"])

encoded = []

for example, label in zip(text, labels):
    encoded += [50257] + enc.encode(example) + [50258]
    if labels == "ham":
        label = 50262
    else:
        label = 50261
    encoded += [50259, label, 50260]

buff = torch.tensor(encoded).to(device)
buff = buff[:1600001]
x = buff[:-1].view(6250, 256)
y = buff[1:].view(6250, 256)

x = x[:64]
y = y[:64]
criteria = nn.CrossEntropyLoss()
pred = model(x)
print(criteria(pred.view(-1, pred.size(-1)), y.view(-1)))


