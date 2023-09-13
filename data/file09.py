from torch import nn
from file07 import EmbeddingBlockFirst
from file08 import EmbeddingBlockSecond

class EmbeddingBlock(nn.Module):
  def __init__(self, vocab_size, d_model, pad_idx, max_steps, dropout=0.1, dtype=None, device=None):
    kargs = { 'dtype': dtype, 'device': device }
    super(EmbeddingBlock, self).__init__()
    self.first = EmbeddingBlockFirst(vocab_size, d_model, pad_idx, max_steps, **kargs)
    self.second = EmbeddingBlockSecond(d_model, max_steps, dropout, **kargs)
  
  def forward(self, X, steps, mask):
    assert X.dim() == 2, 'X must to have shape (batch, seq)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    X_first = self.first(X, steps, mask)
    X_second = self.second(X_first, steps, mask)
    return X_second