import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TinyModel(torch.nn.Module):

  def __init__(self):
    super(TinyModel, self).__init__()

    self.linear1 = nn.Linear(6, 16)
    self.activation = nn.ReLU()
    self.linear2 = nn.Linear(16, 1)
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.softmax(x)
    return x

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layerLast = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layerLast(x)