import torch
from torch import nn

from file03 import create_sin_cos_encoding

class PositionalEncoding(nn.Module):
  def __init__(self, d_model=512, max_seq_length=2048, dtype=None, device=None):
    super(PositionalEncoding, self).__init__()
    self.d_model = d_model
    self.max_seq_length = max_seq_length
    pe = create_sin_cos_encoding(d_model, max_seq_length, dtype, device)
    
    self.register_buffer('pe', pe)
  
  def forward(self, X:torch.Tensor) -> torch.Tensor:
    assert X.dim() == 3, 'X must to have shape (batch, seq, d_model)'
    return X + self.pe[:, :X.shape[1]].to(X.device)

  def __repr__(self):
    return f'{PositionalEncoding.__name__}(d_model={self.d_model}, max={self.max_seq_length})'