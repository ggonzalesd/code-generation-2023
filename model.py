import torch
from torch import nn
import torch.nn.functional as F

class MyEncoderTransformer(nn.Module):
  def __init__(self, d_model, nheads, dropout=0.1, dim_feedforward=2048, activation=F.gelu, device=None, dtype=None):
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super(MyEncoderTransformer, self).__init__()
    
    self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout, False, batch_first=True, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)

    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
    self.dropout2 = nn.Dropout(dropout)

    self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)

    self.activation = activation
  
  def _sa_block(self, x, attn_mask, padding_mask) -> torch.Tensor:
    x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=padding_mask, need_weights=False)
    return self.dropout1(x)

  def _ff_block(self, x) -> torch.Tensor:
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    return self.dropout2(x)

  def forward(self, src, attn_mask=None, padding_mask=None):
    x = src
    x = self.norm1(x + self._sa_block(x, attn_mask, padding_mask))
    x = self.norm2(x + self._ff_block(x))
    return x

class MyTransformer(nn.Module):
  def __init__(self, d_model, nheads, num_layers, dropout=0.1, dim_feedforward=2048, activation=F.gelu, device=None, dtype=None):
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super(MyTransformer, self).__init__()

    self.layers = nn.ModuleList([
      MyEncoderTransformer(d_model, nheads, dropout, dim_feedforward, activation, **factory_kwargs)
      for _ in range(num_layers)
    ])
  
  def forward(self, src, attn_mask=None, padding_mask=None):
    for layer in self.layers:
      src = layer(src, attn_mask, padding_mask)
    return src


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
