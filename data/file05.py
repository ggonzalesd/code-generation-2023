import torch
from torch import nn

from file03 import create_sin_cos_encoding

class StepEncoding(nn.Module):
  def __init__(self, d_model=512, max_seq_length=2048, dtype=None, device=None):
    super(StepEncoding, self).__init__()
    self.d_model = d_model
    self.max_seq_length = max_seq_length
    pe = create_sin_cos_encoding(d_model, max_seq_length, dtype, device)
    
    self.register_buffer('pe', pe)
  
  def forward(self, X:torch.Tensor, steps:torch.Tensor, mask=None) -> torch.Tensor:
    assert X.dim() == 3, 'X must to have shape (batch, seq, d_model)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    if mask is not None: assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    if mask is None:
      return X + self.pe[0][steps,:].unsqueeze(1)
    else:
      steps_per_mask = (mask * steps.unsqueeze(1))[mask]
      X[mask] += self.pe[0][steps_per_mask]
      return X

  def __repr__(self):
    return f'{StepEncoding.__name__}(d_model={self.d_model}, max={self.max_seq_length})'