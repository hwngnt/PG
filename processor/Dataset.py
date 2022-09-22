import torch
import glob
import pandas as pd
import numpy as np
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, label_path):
        'Initialization'
        self.data_path = data_path
        self.label_path = label_path
        self.load_data()
  def load_data(self):
        self.skeleton = np.load(self.data_path)
        self.label = np.load(self.label_path)
      #   print(self.label)
      #   print(self.label.shape, self.skeleton.shape)

  def __len__(self):
        'Denotes the total number of samples'
        return self.label.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        skeleton = self.skeleton[index]
        X = torch.Tensor(skeleton)
        y = torch.Tensor(self.label[index])
        return X, y