import os
import re

import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import pytorch_lightning as pl


def open_image(path):
    return Image.open(path).convert("RGB")


def full_fpath(base_dir, traverse, fname):
    return os.path.join(base_dir, traverse, 'stereo/centre', fname)


class RobotCarSeqDataset(Dataset):
    def __init__(self, img_base_dir, dataset_file):
        self.dataset_file = dataset_file
        self.img_base_dir = img_base_dir
        df = pd.read_csv(dataset_file)
        self.traverse_query = df['traverse_query'].tolist()
        self.traverse_db = df['traverse_db'].tolist()
        self.ts_query = df.filter(regex='ts_query').to_numpy()
        self.ts_db = df.filter(regex='ts_db').to_numpy()
        self.ts_min_err = df.filter(regex='ts_best').to_numpy()
        self.min_err_xy = np.stack((df.filter(regex='best_easting_db_').to_numpy(), df.filter(regex='best_northing_db_').to_numpy()), axis=-1)
        self.fnames_query = df.filter(regex='fname_query_').values.tolist()
        self.fnames_db = df.filter(regex='fname_db_').values.tolist()
        self.xy_query = np.stack((df.filter(regex='easting_query_').to_numpy(), df.filter(regex='northing_query_').to_numpy()), axis=-1)
        self.xy_db = np.stack((df.filter(regex='^easting_db_').to_numpy(), df.filter(regex='^northing_db_').to_numpy()), axis=-1)
        self.xy_db -= self.xy_query[None, None, 0, 0, :]  # utm coordinates are large, causes float precision loss. standardise
        self.min_err_xy -= self.xy_query[None, None, 0, 0, :]
        self.xy_query -= self.xy_query[None, None, 0, 0, :]
        self.max_err = df['max_err'].to_numpy()
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        imgs_db = torch.stack([self.base_transform(open_image(full_fpath(self.img_base_dir, self.traverse_db[idx], fname))) for fname in self.fnames_db[idx]])
        imgs_query = torch.stack([self.base_transform(open_image(full_fpath(self.img_base_dir, self.traverse_query[idx], fname))) for fname in self.fnames_query[idx]])
        ts_query = torch.from_numpy(self.ts_query[idx, :]).float()
        ts_db = torch.from_numpy(self.ts_db[idx, :]).float()
        ts_min_err = torch.from_numpy(self.ts_min_err[idx, :]).float()
        xy_min_err_db = torch.from_numpy(self.min_err_xy[idx, ...]).float()
        xy_query = torch.from_numpy(self.xy_query[idx, ...]).float()
        xy_db = torch.from_numpy(self.xy_db[idx, ...]).float()
        return {'traverse_query': self.traverse_query[idx], 'imgs_db': imgs_db, 'imgs_query': imgs_query, 'ts_query': ts_query, 'ts_db': ts_db,
                'ts_min_err': ts_min_err, 'max_err': self.max_err[idx],
                'min_err_xy': xy_min_err_db, 'xy_query': xy_query, 'xy_db': xy_db, 'fnames_query': self.fnames_query[idx], 'fnames_db': self.fnames_db[idx]}

    def __len__(self):
        return self.ts_query.shape[0]


class DTWDatamodule(pl.LightningDataModule):
    def __init__(self, img_base_dir: str, data_list_dir: str, batch_size_train: int = 8, batch_size_eval: int = 16,
                 num_workers: str = 16, shuffle_train=True):
        super().__init__()
        self.data_list_dir = data_list_dir
        self.img_base_dir = img_base_dir
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

    def setup(self, stage: str = None):
        self.train_seqs_full = RobotCarSeqDataset(self.img_base_dir, os.path.join(self.data_list_dir, 'sequence_pairs_train.csv'))
        self.train_seqs = self.train_seqs_full
        self.val_seqs = RobotCarSeqDataset(self.img_base_dir, os.path.join(self.data_list_dir, 'sequence_pairs_validation.csv'))
        self.test_seqs = [RobotCarSeqDataset(self.img_base_dir, os.path.join(self.data_list_dir, f)) for f in os.listdir(self.data_list_dir)
                          if re.match(r'sequence_pairs_test_.*.csv', f)]

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_seqs, batch_size=self.batch_size_train, num_workers=self.num_workers, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.val_seqs, batch_size=self.batch_size_eval, num_workers=self.num_workers)

    def test_dataloader(self):
        return [DataLoader(test_seqs, batch_size=self.batch_size_eval, num_workers=self.num_workers) for test_seqs in self.test_seqs]

