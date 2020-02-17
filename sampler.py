import numpy as np
from torch.utils.data import Sampler

class CustomSampler(Sampler):
  def __init__(self, data_source, shuffle):
    self.data_source = data_source
    self.df = self.data_source.wavfiles_frame
    self.shuffle = shuffle

  def getIndices(self):
    labels = np.unique(self.df['target']).tolist()
    digit_indices = [np.where(self.df['target'] == i)[0] for i in labels]
    if self.shuffle:
      for i in range(len(digit_indices)):
        np.random.shuffle(digit_indices[i])
    min_size = np.size(digit_indices[0])
    for i in range(1, len(digit_indices)):
      size = np.size(digit_indices[i])
      min_size = size if size < min_size else min_size
    return digit_indices, min_size

  def __iter__(self):
    digit_indices, min_size = self.getIndices()
    num_classes = len(digit_indices)
    indices = []
    for i in range(min_size):
      indices += [digit_indices[n][i] for n in range(num_classes)]
    return iter(indices)

  def __len__(self):
    digit_indices, min_size = self.getIndices()
    return min_size * len(digit_indices)