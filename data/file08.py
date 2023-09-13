from torch import nn

from file04 import PositionalEncoding
from file05 import StepEncoding

class EmbeddingBlockSecond(nn.Module):
  def __init__(self, d_model, max_steps, dropout=0.1, dtype=None, device=None):
    kargs = { 'dtype': dtype, 'device': device }
    super(EmbeddingBlockSecond, self).__init__()
    self.pe = PositionalEncoding(d_model, 512, **kargs)
    self.se = StepEncoding(d_model, max_steps, **kargs)
    self.norm = nn.LayerNorm(d_model, **kargs)
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