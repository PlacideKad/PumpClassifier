
import torch.nn as nn
class PumpClassifier(nn.Module):
  def __init__(self):
    super(PumpClassifier,self).__init__()
    self.model=nn.Sequential(
      nn.Linear(4,64),
      nn.ReLU(),
      nn.Linear(64,128),
      nn.ReLU(),
      nn.Linear(128,64),
      nn.ReLU(),
      nn.Linear(64,20)
    )
  def forward(self,x):
    return self.model(x)