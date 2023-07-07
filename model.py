import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork,self).__init__()

        # 3 numbers of Layers
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.lin2 = nn.Linear(hidden_size,hidden_size)
        self.lin3 = nn.Linear(hidden_size,num_classes)
        
        self.relu = nn.ReLU() # Activation function in between the layers ReLU

    # Implement the Forward Pass in NN
    def forward (self,x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        # No need of Activation fuction to the softmax the last layer
        return out

#

