import torch
from torch import nn
from file06 import MaskedForwardDiffusion

class EmbeddingBlockFirst(nn.Module):
  def __init__(self, vocab_size, d_model, pad_idx, max_steps, dtype=None, device=None):
    kargs = { 'dtype': dtype, 'device': device }
    super(EmbeddingBlockFirst, self).__init__()
    self.word_embedding = nn.Embedding(vocab_size, d_model, pad_idx, **kargs)
    self.masked_forward_diffusion = MaskedForwardDiffusion(max_steps, **kargs)

  def forward(self, X, steps, mask):
    assert X.dim() == 2, 'X must to have shape (batch, seq)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    embed = self.word_embedding(X)
    embed_diff = self.masked_forward_diffusion(embed, steps, mask)
    return embed_diff