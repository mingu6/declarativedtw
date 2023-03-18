import os
import os.path as path
import math
import re
from itertools import accumulate
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


# used to rescale features before training (mu, sigma)
rescale_params = {
    'cqt': (-0.1380, 1.),
    'chroma': (-12.368, 45.),
    'melspec': (-20.000, 268.)
}


def map_score(perf):
    """ taken from alignment-eval repo for convenience """
    """ associate a performance midi with a kern score based on filename conventions """
    regex = re.compile('(\d\d\d)_bwv(\d\d\d)(f|p)')
    info = regex.search(perf)
    num, bwv, part = info.group(1,2,3)
    bwv = int(bwv)
    book = 1 + int(bwv > 869)
    score = 'wtc{}{}{:02d}'.format(book,part,bwv - 845 - (book-1)*24)

    return score


class SlicedMusicDataset(Dataset):
    def __init__(self, base_path: str, feat_type: str, slice_len: int = 256, perf_fnames=None, rng=None):
        align_path = path.join(base_path, 'align', 'ground')
        score_path = path.join(base_path, 'data', 'score_feats', feat_type)
        perf_path = path.join(base_path, 'data', 'perf_feats', feat_type)
        self.perf_fnames = sorted(os.listdir(perf_path)) if perf_fnames is None else perf_fnames
        self.slice_len = slice_len
        self.rng = rng
        # read ground truth alignments
        self.gt = [np.loadtxt(path.join(align_path, perf_fname[:-3] + 'txt')) for perf_fname in self.perf_fnames]
        self.gt = [torch.from_numpy(arr.astype(np.float32)) for arr in self.gt]
        # read all features to RAM
        perf_data = [np.load(path.join(perf_path, perf_fname[:-3] + 'npz')) for perf_fname in self.perf_fnames]
        score_data = [np.load(path.join(score_path, perf_fname[:-3] + 'npz')) for perf_fname in self.perf_fnames]
        self.perf_times = [z['times'].astype(np.float32) for z in perf_data]
        self.score_times = [z['times'].astype(np.float32) for z in score_data]
        self.perf_feats = [z['feats'].astype(np.float32) for z in perf_data]
        self.score_feats = [z['feats'].astype(np.float32) for z in score_data]
        # rescale (roughly standardise) features
        mu, sigma = rescale_params[feat_type]
        self.perf_feats = [(z - mu) / sigma for z in self.perf_feats]
        self.score_feats = [(z - mu) / sigma for z in self.score_feats]
        # generate equal size slices from score/performance for train/eval
        self.score_lengths = [len(t) for t in self.score_times]
        dt_audio = self.perf_times[0][1] - self.perf_times[0][0]
        self.slices = self._generate_slices(slice_len, dt_audio)

    def _generate_slices(self, slice_len: int, dt_audio: float):
        slices = []
        tot_slices = 0
        score_slc_dur = slice_len * dt_audio  # duration per slice w.r.t. score midi clip
        perf_slc_dur = score_slc_dur * 1.1  # interval for performance (larger in case of tempo deviations)
        perf_slc_len = math.ceil(perf_slc_dur / dt_audio)
        for i, perf in enumerate(self.perf_fnames):
            score_gt_t, perf_gt_t = self.gt[i].T
            score_slc_start_times = np.arange(0., score_gt_t[-1], score_slc_dur)  # generate slice times of length slice_len
            for sc_lb, sc_ub in zip(score_slc_start_times[:-1], score_slc_start_times[1:]):
                # within each slice, locate relevant gt alignment observations
                gt_lb_ind = max(np.searchsorted(score_gt_t, sc_lb, side='right'), 0)
                gt_ub_ind = np.searchsorted(score_gt_t, sc_ub) - 1
                pf_gt_lb, pf_gt_ub = perf_gt_t[gt_lb_ind], perf_gt_t[gt_ub_ind]
                # ensure performance is contained in sliced rectangle (tempo deviations not too large)
                if pf_gt_ub - pf_gt_lb < perf_slc_dur and gt_lb_ind != gt_ub_ind:
                    # locate indices to slice full audio clips for each slice
                    sc_audio_lb_ind = int(sc_lb / dt_audio)
                    sc_audio_ub_ind = sc_audio_lb_ind + slice_len
                    perf_margin = pf_gt_lb + perf_slc_dur - pf_gt_ub
                    offset = np.random.uniform(0., perf_margin) / 2 if self.rng is None else self.rng.uniform(0., perf_margin) / 2
                    pf_audio_lb_ind = max(int((pf_gt_lb - offset) / dt_audio), 0)
                    pf_audio_ub_ind = pf_audio_lb_ind + perf_slc_len
                    if pf_audio_ub_ind * dt_audio < perf_gt_t[-1] and pf_audio_ub_ind <= len(self.perf_times[i]):
                        slices.append((i, sc_audio_lb_ind, sc_audio_ub_ind, pf_audio_lb_ind, pf_audio_ub_ind, gt_lb_ind, gt_ub_ind))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):
        i, l_s, u_s, l_p, u_p, l_g, u_g = self.slices[item]  # return i
        # slice perf and score feats and times
        score_times = self.score_times[i][l_s:u_s]
        perf_times = self.perf_times[i][l_p:u_p]
        score_feats = self.score_feats[i][l_s:u_s]
        perf_feats = self.perf_feats[i][l_p:u_p]
        return torch.LongTensor([i]), torch.from_numpy(score_times).clone(), torch.from_numpy(perf_times).clone(), torch.from_numpy(score_feats).clone(), \
               torch.from_numpy(perf_feats).clone(), torch.LongTensor([l_g]), torch.LongTensor([u_g])


class MusicDM(pl.LightningDataModule):
    def __init__(self, base_path: str, feat_type: str, slice_len: int, batch_size: int, seed=1,
                 num_workers=0, splits: Tuple[float, float, float] = (0.5, 0.25, 0.25)):
        super().__init__()
        self.base_path = base_path
        self.feat_type = feat_type
        self.slice_len = slice_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits
        self.rng = np.random.RandomState(seed=seed)
        # need score
        train, val, test = self.splits
        align_path = path.join(self.base_path, 'align', self.feat_type)
        perf_fnames = sorted(os.listdir(align_path))
        fnames_train, fnames_val, fnames_test = self._split_perfs_by_score(perf_fnames, splits)
        self.train = SlicedMusicDataset(self.base_path, self.feat_type, self.slice_len, perf_fnames=fnames_train, rng=self.rng)
        self.val = SlicedMusicDataset(self.base_path, self.feat_type, self.slice_len, perf_fnames=fnames_val, rng=self.rng)
        self.test = SlicedMusicDataset(self.base_path, self.feat_type, self.slice_len, perf_fnames=fnames_test, rng=self.rng)

    def _split_perfs_by_score(self, perf_fnames, splits: Tuple[float, float, float]):
        train, val, test = splits
        # identify a mapping between scores to performances (one-to-many)
        score_names = [map_score(perf) for perf in perf_fnames]
        scores_uniq = sorted(list(set(score_names)))
        score_to_perfs = defaultdict(list)
        for sc, pf in zip(score_names, perf_fnames):
            score_to_perfs[sc].append(pf)
        # shuffle scores and select splits s.t. desired proportions w.r.t. no. performances are met
        score_count = [len(score_to_perfs[sc]) for sc in scores_uniq]
        ind_shuffle = self.rng.permutation(len(score_count))
        score_count = list(accumulate([score_count[i] for i in ind_shuffle]))
        scores_uniq = [scores_uniq[i] for i in ind_shuffle]
        # find split indices
        ind_train_max = next((i for i, cnt in enumerate(score_count) if cnt > train * score_count[-1]), None)
        ind_val_max = next((i for i, cnt in enumerate(score_count) if cnt > (train + val) * score_count[-1]), None)
        # recover performance fnames by looping through scores and applying score -> perfs mapping
        perfs_train = []
        for sc in scores_uniq[:ind_train_max]:
            perfs_train.extend(score_to_perfs[sc])
        perfs_val = []
        for sc in scores_uniq[ind_train_max:ind_val_max]:
            perfs_val.extend(score_to_perfs[sc])
        perfs_test = []
        for sc in scores_uniq[ind_val_max:]:
            perfs_test.extend(score_to_perfs[sc])
        return perfs_train, perfs_val, perfs_test

    def setup(self, stage: Optional[str] = None) -> None:
        return None

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

