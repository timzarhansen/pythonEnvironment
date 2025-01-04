from gptmodel import GPTLanguageModel
import torch
import torch.nn as nn
from torch.nn import functional as F


class SubModel(GPTLanguageModel):
    def __init__(self):
        super(SubModel, self).__init__()
        self.layer3 = nn.Linear(30, 40)  # Add a new layer

    # def forward(self, x):
    #     x = super().forward(x)
    #     x = super().lm_head
    #     x = self.layer3(x)
    #     return x
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = super().token_embedding_table(idx) # (B,T,C)
        pos_emb = super().position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = super().blocks(x) # (B,T,C)
        x = super().ln_f(x) # (B,T,C)
        logits = super().lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


