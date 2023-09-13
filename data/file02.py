import torch
from torch import nn
import torch.nn.functional as F

class Classification(nn.Module):
  def __init__(self, d_model, vocab_size, dropout=0.1, dim_feedforward=2048, activation=F.gelu, device=None, dtype=None):
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super(Classification, self).__init__()

    self.linear = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.linear_out = nn.Linear(dim_feedforward, vocab_size, **factory_kwargs)

  def forward(self, X):
    X = self.activation(self.dropout(self.linear(X)))
    return self.linear_out(X)
