import time
from lladaModel import LLaDAModel
from configuration_llada import ModelConfig
import torch
import torch.nn as nn
from torch.nn import functional as F

from pythonEnvironment.MachineLearning.largelanguagediffusionmodel import configuration_llada

#1414 s mit MPS
# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 200
learning_rate = 1e-4
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200 #was 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------




# data loading
def get_batch(split,train_data,val_data,maskedTokenID):
    # generate a small batch of data of inputs x and targets y
    t = torch.rand(batch_size) #what percentage should be masked of the message?


    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    #masking
    mask = torch.rand(batch_size,block_size) < t.unsqueeze(1)  # Compare random values against per-element thresholds in `t`
    x[mask] = maskedTokenID  # Replace elements where mask is True
    y = torch.stack([data[i:i+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model,train_data,val_data,maskedTokenID):
    # criterion = nn.CrossEntropyLoss()
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data,maskedTokenID)
            outputModel = model(X)
            mask = (X == maskedTokenID)
            # masked_outputs = X[==maskedTokenID]
            # masked_labels = labels[mask]
            mask = mask.bool()  # Ensure boolean type if not already
            # loss = F.cross_entropy(outputModel.logits[mask], Y[mask])
            Y_masked = Y[mask]
            output_models_masked = outputModel.logits[mask,:]
            # loss = F.cross_entropy(output_models_masked.view(-1, output_models_masked.size(-1)), Y_masked.view(-1))
            loss = F.cross_entropy(output_models_masked, Y_masked)
            #compute loss stuff
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('../data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    chars.append('_')
    vocab_size = len(chars) # for the masked token +1
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    config = configuration_llada.ModelConfig()
    config.init_device = device  # meta mps cpu
    config.vocab_size = vocab_size
    config.embedding_size = None
    config.max_sequence_length = block_size  # context length
    config.n_layers = n_layer
    config.n_heads = n_head
    config.d_model = n_embd
    config.rope = True
    config.attention_dropout = dropout

    model = LLaDAModel(config)
    print(model)
    model.load_state_dict(torch.load('../modelSaved/diffusionModel.pth'))
    model.to(device)
    model.eval()



    # generate from the model
    contextInput = torch.zeros((1, block_size), dtype=torch.long, device=device)
    contextInput.fill_(encode('_')[0])

    ix = torch.randint(len(train_data) - block_size, (1,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    # sampleInput =
    contextInput[0,0:50]= x[0,0:50]



    sampled_idx = contextInput
    sampled_idx = sampled_idx.view(-1)
    print(decode(sampled_idx.tolist()))
    print("---------------------------------------------------------------------------")

    for i in range(10):
        llmOutput = model(sampled_idx.view(1,-1))
        probs = torch.softmax(llmOutput.logits, dim=-1).view(block_size,vocab_size)
        sampled_idx = torch.multinomial(probs, 1).view(block_size)
        print(decode(sampled_idx.tolist()))

        # masking
        mask = torch.rand(block_size) < 0.1  # 10 percent is changed to token
        sampled_idx[mask] = encode('_')[0]  # Replace elements where mask is True
        sampled_idx[0:50] = x[0, 0:50]
        print(decode(sampled_idx.tolist()))
        print("---------------------------------------------------------------------------")





    # print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    # torch.save(model.state_dict(), '../modelSaved/diffusionModel.pth')
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))