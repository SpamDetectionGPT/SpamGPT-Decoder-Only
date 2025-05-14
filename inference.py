import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import matplotlib.pyplot as plt

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

import math
import numpy as np
import matplotlib.pyplot as plt

def get_spam_ham_probabilities(input_tokens, model, temp=1):
    """
    Returns spam_prob, ham_prob, highest_prob_token
    and saves a grid of attention maps (one per head) as attention_matrix.png
    """
    model.eval()
    with torch.no_grad():
        # ----------  spam / ham logic (unchanged) -----------------------
        pred   = model(input_tokens)
        logits = pred[:, -1, :] / temp
        all_p  = F.softmax(logits, dim=-1)

        highest_token = enc.decode([torch.argmax(all_p).item()])

        spam_ham_logits = logits[0, [special_tokens["<SPAM>"],
                                     special_tokens["<HAM>"]]]
        probs = F.softmax(spam_ham_logits.unsqueeze(0), dim=-1)
        spam_prob, ham_prob = probs[0].tolist()
        # ----------------------------------------------------------------

        # ----------  grab the whole attention tensor --------------------
        att_tensor = model.transformer['h'][-1].attn.latest_attn[0]   # (H, T, T)
        n_head, T, _ = att_tensor.shape
        # ----------------------------------------------------------------

        # ----------  decode tokens for tick labels ----------------------
        tok_ids      = input_tokens[0].tolist()
        tick_labels  = []
        for tid in tok_ids:
            if tid in special_tokens.values():
                tick_labels.append(next(k for k,v in special_tokens.items() if v == tid))
            else:
                tick_labels.append(enc.decode([tid]).replace("\n", "\\n"))
        # ----------------------------------------------------------------

        # ----------  plotting all heads ---------------------------------
        cols = 3                               # 2 × 3 grid for 6 heads
        rows = int(math.ceil(n_head / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols*4, rows*4),
                                 squeeze=False)

        for h in range(n_head):
            r, c = divmod(h, cols)
            ax   = axes[r][c]
            im   = ax.imshow(att_tensor[h].cpu().numpy())
            ax.set_title(f"Head {h}", fontsize=10)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
            ax.set_yticklabels(tick_labels, fontsize=6)

        # turn off any empty panes (if n_head is not a multiple of cols)
        for h in range(n_head, rows*cols):
            r, c = divmod(h, cols)
            axes[r][c].axis("off")

        fig.suptitle("Last-Layer Attention Maps", fontsize=12)
        plt.tight_layout()
        fig.savefig("attention_matrix.png", dpi=300)
        plt.close(fig)
        # ----------------------------------------------------------------

        return spam_prob, ham_prob, highest_token

