import math

import torch
from torch import nn
import torch.nn.functional as F

def create_sin_cos_encoding(d_model, max_seq_length, dtype, device):
  factory_kwargs = {
    'dtype': dtype,
    'device': device
  }
  pe = torch.zeros(max_seq_length, d_model, **factory_kwargs)
  position = torch.arange(0, max_seq_length, **factory_kwargs).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2, **factory_kwargs).float() * -(math.log(1e+4) / d_model))

  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)

  return pe.unsqueeze(0)

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

class MaskedForwardDiffusion(nn.Module):
  def __init__(self, max_steps, dtype=None, device=None):
    self.factory_kwargs = {
      'dtype': dtype,
      'device': device
    }
    super(MaskedForwardDiffusion, self).__init__()
    self.max_steps = max_steps

  def forward(self, X, steps, mask):
    assert X.dim() == 3, 'X must to have shape (batch, seq, d_model)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    noise = torch.randn_like(X, **self.factory_kwargs)

    noise_intense = 1.0 - (steps / self.max_steps).unsqueeze(-1)
    noise_intense = 1.0 - torch.cos(torch.pi * noise_intense / 2)

    X[mask] *= (mask * noise_intense)[mask].unsqueeze(-1)
    X[mask] += noise[mask] * (mask * (1.0-noise_intense))[mask].unsqueeze(-1)

    return X

  def __repr__(self):
    return f"{MaskedForwardDiffusion.__name__}(steps={self.max_steps})"

class EmbeddingBlockFirst(nn.Module):
  def __init__(self, vocab_size, d_model, pad_idx, max_steps, dtype=None, device=None):
    factory_kwargs = { 'dtype': dtype, 'device': device }
    super(EmbeddingBlockFirst, self).__init__()
    self.word_embedding = nn.Embedding(vocab_size, d_model, pad_idx, **factory_kwargs)
    self.masked_forward_diffusion = MaskedForwardDiffusion(max_steps, **factory_kwargs)

  def forward(self, X, steps, mask):
    assert X.dim() == 2, 'X must to have shape (batch, seq)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    embed = self.word_embedding(X)
    embed_diff = self.masked_forward_diffusion(embed, steps, mask)
    return embed_diff

class EmbeddingBlockSecond(nn.Module):
  def __init__(self, d_model, max_steps, dropout=0.1, dtype=None, device=None):
    factory_kwargs = { 'dtype': dtype, 'device': device }
    super(EmbeddingBlockSecond, self).__init__()
    self.pe = PositionalEncoding(d_model, 512, **factory_kwargs)
    self.se = StepEncoding(d_model, max_steps, **factory_kwargs)
    self.norm = nn.LayerNorm(d_model, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, X, steps, mask):
    assert X.dim() == 3, 'X must to have shape (batch, seq, d_model)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    X_pe = self.pe(X)
    X_se = self.se(X_pe, steps, mask)
    norm = self.norm(X_se)
    out = self.dropout(norm)
    return out

class EmbeddingBlock(nn.Module):
  def __init__(self, vocab_size, d_model, pad_idx, max_steps, dropout=0.1, dtype=None, device=None):
    factory_kwargs = { 'dtype': dtype, 'device': device }
    super(EmbeddingBlock, self).__init__()
    self.first = EmbeddingBlockFirst(vocab_size, d_model, pad_idx, max_steps, **factory_kwargs)
    self.second = EmbeddingBlockSecond(d_model, max_steps, dropout, **factory_kwargs)
  
  def forward(self, X, steps, mask):
    assert X.dim() == 2, 'X must to have shape (batch, seq)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    X_first = self.first(X, steps, mask)
    X_second = self.second(X_first, steps, mask)
    return X_second

if __name__ == '__main__':
  d_model = 128
  data = torch.randn(2, 7, d_model)

  mask = torch.BoolTensor([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1]
  ])
  steps = torch.LongTensor([5, 2])

  pe = PositionalEncoding(d_model)
  se = StepEncoding(d_model)

  pe(data)
  se(data, steps, mask)