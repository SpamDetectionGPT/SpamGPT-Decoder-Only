import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken

base_encoding = tiktoken.get_encoding("gpt2")
special_tokens = {
    "<SOE>": 50257,  # Add one after the vocab size (GPT-2 vocab size is 50257)
    "<EOE>": 50258,
    "<SOP>": 50259,
    "<EOP>": 50260,
    "<SPAM>": 50261,
    "<HAM>": 50262,
}
enc = tiktoken.Encoding(
    name="gpt2-custom",
    pat_str=base_encoding._pat_str,
    mergeable_ranks=base_encoding._mergeable_ranks,
    special_tokens={**base_encoding._special_tokens, **special_tokens}
)


def run_inference(input_tokens, max_length, model, temp, enc, endtoken=None):
    model.eval()
    print("\n\n-----------------Inference-----------------\n")
    while input_tokens.size()[1] < max_length:
        if endtoken is not None and input_tokens[0, -1].item() == endtoken:
            break
        pred = model(input_tokens)
        logits = pred[:,-1, :]
        logits = logits/temp
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        print(enc.decode([next_token.squeeze().tolist()]), end="")
        input_tokens = torch.cat((input_tokens, next_token), dim=1)
    print("\n\n---------------End Inference---------------\n")
    return input_tokens