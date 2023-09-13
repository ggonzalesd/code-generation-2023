import torch
from torch import nn

class MaskedForwardDiffusion(nn.Module):
  def __init__(self, max_steps, dtype=None, device=None):
    self.kargs = {
      'dtype': dtype,
      'device': device
    }
    super(MaskedForwardDiffusion, self).__init__()
    self.max_steps = max_steps

  def forward(self, X, steps, mask):
    assert X.dim() == 3, 'X must to have shape (batch, seq, d_model)'
    assert steps.dim() == 1, 'steps must to have shape (batch)'
    assert mask.dim() == 2, 'mask must to have shape (batch, seq)'

    noise = torch.randn_like(X, **self.kargs)

    noise_intense = 1.0 - (steps / self.max_steps).unsqueeze(-1)
    noise_intense = 1.0 - torch.cos(torch.pi * noise_intense / 2)

    X[mask] *= (mask * noise_intense)[mask].unsqueeze(-1)
    X[mask] += noise[mask] * (mask * (1.0-noise_intense))[mask].unsqueeze(-1)

    return X

  def __repr__(self):
    return f"{MaskedForwardDiffusion.__name__}(steps={self.max_steps})"