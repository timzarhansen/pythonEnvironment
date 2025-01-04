from gptmodel import GPTLanguageModel
import torch
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2




with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string



model = GPTLanguageModel(vocab_size, n_head, n_embd, n_layer, block_size, device, dropout)
model.load_state_dict(torch.load('modelSaved/gptModel.pth'))
model.to(device)
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)

for param in model.parameters():
    param.requires_grad = False


# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
