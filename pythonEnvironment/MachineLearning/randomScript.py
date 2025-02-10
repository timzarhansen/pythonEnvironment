import torch
import torch.nn as nn
from recursiveGPTModel import RecursiveGPTModel

device = 'mps' if torch.mps.is_available() else 'cpu'


with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


pretrained_dict = torch.load('modelSaved/gptModel.pth')

# Instantiate the SubModel
submodel = RecursiveGPTModel(vocab_size,device)

# Load the state dict, but only for the layers that match
submodel.load_state_dict(pretrained_dict, strict=False)
submodel.to(device)
submodel.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(submodel.generate(context, max_new_tokens=500)[0].tolist()))





