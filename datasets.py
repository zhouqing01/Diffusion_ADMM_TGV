# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


from pathlib import Path

class fastmri_knee(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root, is_complex=False):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))
    self.is_complex = is_complex

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    if not self.is_complex:
      data = np.load(fname)
    else:
      data = np.load(fname).astype(np.complex64)
    data = np.expand_dims(data, axis=0)
    return data


class AAPM(Dataset):
  def __init__(self, root, sort):
    self.root = root
    self.data_list = list(root.glob('full_dose/*.npy'))
    self.sort = sort
    if sort:
      self.data_list = sorted(self.data_list)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname)
    data = np.expand_dims(data, axis=0)
    return data


class Object5(Dataset):
  def __init__(self, root, slice, fast=False):
    """
    slice - range of the 2000 _volumes_ that you want,
    but the dataset will return images, so will be 256 times longer

    fast - set to true to get a tiny version of the dataset
    """
    if fast:
      self.NUM_SLICES = 10
    else:
      self.NUM_SLICES = 256


    self.root = root
    self.data_list = list(root.glob('*.npz'))

    if len(self.data_list) == 0:
      raise ValueError(f"No npz files found in {root}")

    self.data_list = sorted(self.data_list)[slice]

  def __len__(self):
    return len(self.data_list) * self.NUM_SLICES

  def __getitem__(self, idx):
    vol_index = idx // self.NUM_SLICES
    slice_index = idx % self.NUM_SLICES
    fname = self.data_list[vol_index]
    data = np.load(fname)['x'][slice_index]
    data = np.expand_dims(data, axis=0)
    return data

class fastmri_knee_infer(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root, sort=True, is_complex=False):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))
    self.is_complex = is_complex
    if sort:
      self.data_list = sorted(self.data_list)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    if not self.is_complex:
      data = np.load(fname)
    else:
      data = np.load(fname).astype(np.complex64)
    data = np.expand_dims(data, axis=0)
    return data, str(fname)


class fastmri_knee_magpha(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname).astype(np.float32)
    return data


class fastmri_knee_magpha_infer(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root, sort=True):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))
    if sort:
      self.data_list = sorted(self.data_list)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname).astype(np.float32)
    return data, str(fname)


def create_dataloader(configs, evaluation=False, sort=True):
  shuffle = True if not evaluation else False
  if configs.data.dataset == 'Object5':
    train_dataset = Object5(Path(configs.data.root), slice(None,1800))  
    val_dataset = Object5(Path(configs.data.root), slice(1800,None)) 
  elif configs.data.dataset == 'Object5Fast':
    train_dataset = Object5(Path(configs.data.root), slice(None,1), fast=True)
    val_dataset = Object5(Path(configs.data.root), slice(1,2), fast=True)
  elif configs.data.dataset == 'AAPM':
    train_dataset = AAPM(Path(configs.data.root) / f'train', sort=False)
    val_dataset = AAPM(Path(configs.data.root) / f'test', sort=True)
  elif configs.data.is_multi:
    train_dataset = fastmri_knee(Path(configs.data.root) / f'knee_multicoil_{configs.data.image_size}_train')
    val_dataset = fastmri_knee_infer(Path(configs.data.root) / f'knee_{configs.data.image_size}_val', sort=sort)
  elif configs.data.is_complex:
    if configs.data.magpha:
      train_dataset = fastmri_knee_magpha(Path(configs.data.root) / f'knee_complex_magpha_{configs.data.image_size}_train')
      val_dataset = fastmri_knee_magpha_infer(Path(configs.data.root) / f'knee_complex_magpha_{configs.data.image_size}_val')
    else:
      train_dataset = fastmri_knee(Path(configs.data.root) / f'knee_complex_{configs.data.image_size}_train', is_complex=True)
      val_dataset = fastmri_knee_infer(Path(configs.data.root) / f'knee_complex_{configs.data.image_size}_val', is_complex=True)
  elif configs.data.dataset == 'fastmri_knee':
    train_dataset = fastmri_knee(Path(configs.data.root) / f'knee_{configs.data.image_size}_train')
    val_dataset = fastmri_knee_infer(Path(configs.data.root) / f'knee_{configs.data.image_size}_val', sort=sort)
  else:
    raise ValueError(f'Dataset {configs.data.dataset} not recognized.')

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=configs.training.batch_size,
    shuffle=shuffle,
    drop_last=True
  )
  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=configs.training.batch_size,
    # shuffle=False,
    shuffle=True,
    drop_last=True
  )
  return train_loader, val_loader



def create_dataloader_regression(configs, evaluation=False):
  shuffle = True if not evaluation else False
  train_dataset = fastmri_knee(Path(configs.root) / f'knee_{configs.image_size}_train')
  val_dataset = fastmri_knee_infer(Path(configs.root) / f'knee_{configs.image_size}_val')

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=configs.batch_size,
    shuffle=shuffle,
    drop_last=True
  )
  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=configs.batch_size,
    shuffle=False,
    drop_last=True
  )
  return train_loader, val_loader
