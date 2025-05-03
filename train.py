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


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = SpamGPT(SpamGPTConfig()).to(device)

with open("combined_spam.json", "rb") as h:
    ham = json.load(h)

print(ham[:50])

"""
encoded = []

for example, label in zip(text, labels):
    encoded += [50257] + enc.encode(example) + [50258]
    if labels == "ham":
        label = 50262
    else:
        label = 50261
    encoded += [50259, label, 50260]

buff = torch.tensor(encoded).to(device)
buff = buff[:1638401]
input_ = buff[:-1].view(6400, 256)
labels_ = buff[1:].view(6400, 256)


criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(120):
    for batches in range(100):
        x = input_[64*batches:(64 + 64*batches)]
        y = labels_[64*batches:(64 + 64*batches)]
        pred = model(x)
        loss = criteria(pred.view(-1, pred.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        loss_val = loss.item()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss_val}")

    input_id = torch.tensor([[50257]], device=device, dtype=torch.long)
    output_ids = generate(model, input_id, max_new_tokens=50, temperature=0.8, top_k=50)[0]
    print(enc.decode(output_ids.tolist()))


"""




