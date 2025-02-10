import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from recursiveGPTModel  import SubModel



pretrained_dict = torch.load('modelSaved/gptModel.pth')

# Instantiate the SubModel
submodel = SubModel()

# Load the state dict, but only for the layers that match
submodel.load_state_dict(pretrained_dict, strict=False)








