import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

class DataPy(Dataset):
  def __init__(self, csv_filename, path, transform=None):
    super(DataPy, self).__init__()
    self.df = pd.read_csv(csv_filename).sample(frac=1).reset_index(drop=True)
    self.path = path
    self.transform = transform
  
  def __len__(self):
    return len(self.df)

  def _get_path(self, index):
    filename = self.df.iloc[index].filename
    path = os.path.join(self.path, filename)
    return path
  
  def _read_file(self, path):
    with open(path, 'r') as f:
      data = f.read()
    return data

  def __getitem__(self, index):
    path = self._get_path(index)
    data = self._read_file(path)
    if self.transform is not None:
      data = self.transform(data)
    return data

  def __repr__(self):
    return f"<DataPy len:{len(self)} path:{self.path}>"

  @staticmethod
  def create_collate_fn(batch_first=True, padding_value=0):
    def collate_fn(batch):
      X = nn.utils.rnn.pad_sequence(batch, batch_first=batch_first, padding_value=padding_value)
      length = torch.LongTensor([ a.shape[0] for a in batch])
      padding_mask = X == padding_value
      return X, length, padding_mask
    return collate_fn

class PreProcessingTransform(nn.Module):
  def __init__(self, tokenizer):
    super(PreProcessingTransform, self).__init__()
    self.tokenizer = tokenizer
  def forward(self, text):
    return self.tokenizer.encode(text, return_tensors='pt')[0]