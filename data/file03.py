import math

import torch
from torch import nn
import torch.nn.functional as F

def create_sin_cos_encoding(d_model, max_seq_length, dtype, device):
  kargs = {
    'dtype': dtype,
    'device': device
  }
  pe = torch.zeros(max_seq_length, d_model, **kargs)
  position = torch.arange(0, max_seq_length, **kargs).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2, **kargs).float() * -(math.log(1e+4) / d_model))

  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)

  return pe.unsqueeze(0)