import argparse
import copy

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import sys
sys.path.append('../..')
from decdtw.utils import BatchedSignal
from decdtw.decdtw import DecDTWLayer
from datasets import MusicDM
import comparisons

# torch.use_deterministic_algorithms(True)

backbones = {
    'GRU': lambda d_in, d_out, n_layers: nn.GRU(d_in, int(d_out / 2), bidirectional=True, num_layers=n_layers, batch_first=True),
    'None': lambda d_in, d_out, n_layers: nn.Identity(d_in)
}


class MusicDTWNet(pl.LightningModule):
    def __init__(self, feature_type: str, loss_fn: str, lr: float, reg_wt: float, optimiser: str = 'adam', out_dim: int = 128,
                 num_layers: int = 2, encoder: str = 'GRU', n_warp_discr: int = 1024, mem_efficient: bool = True, **kwargs):
        super().__init__()
        self.feat_type = feature_type
        feat_dim = {'cqt': 48, 'chroma': 12, 'melspec': 128}
        self.feat_dim = feat_dim[feature_type]
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimiser = optimiser
        self.mem_efficient = mem_efficient
        self.encoder = encoder
        self.backbone_perf = backbones[encoder](self.feat_dim, out_dim, num_layers)
        self.backbone_score = copy.deepcopy(self.backbone_perf)
        # save both classic dtw and gdtw for test evaluation
        self.gdtw = DecDTWLayer(mem_efficient=mem_efficient, subseq_enabled=True, n_warp_discr=n_warp_discr)
        self.classicdtw = comparisons.DTWLayer(subseq=False)
        # loss function used during training
        if loss_fn in ['time_err', 'time_dev']:
            self.dtw = self.gdtw
        elif loss_fn == 'dtwnet':
            self.dtw = self.classicdtw
        elif loss_fn == 'softdtw':
            self.dtw = comparisons.SoftDTWLayer(subseq=False, gamma=kwargs['gamma'])
        elif loss_fn == 'dilate':
            self.dtw = comparisons.DILATE(subseq=False, alpha=0., gamma=kwargs['gamma'])
        else:
            self.dtw = comparisons.L2AlongGT(subseq=False)
        self.reg_wt = reg_wt
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.optimiser == 'sgd':
            optimizer = torch.optim.SGD([{'params': self.backbone_perf.parameters()}, {'params': self.backbone_score.parameters()}], lr=self.lr)
        elif self.optimiser == 'adam':
            optimizer = torch.optim.Adam([{'params': self.backbone_perf.parameters()}, {'params': self.backbone_score.parameters()}], lr=self.lr)
        else:
            raise ValueError("invalid optimiser selected, valid choices include 'sgd', 'adam'.")
        return optimizer

    def training_step(self, batch, batch_idx):
        i_b, s_t, p_t, s_f, p_f, l_g, u_g = batch
        s_f, p_f = self._extract_features(batch, batch_idx)
        gt_align = self.trainer.datamodule.train.gt
        with torch.no_grad():
            gt_warps = [self._ground_truth_warp(i, l, u, gt_align, s_t1, p_t1, p_f.device) for
                        i, l, u, s_t1, p_t1 in zip(i_b, l_g, u_g, s_t, p_t)]
        w_t, w_v, dtw_cost = self._predict_warp(s_f, p_f, s_t, p_t, gt_warps)

        time_err = self._time_err(w_t, w_v, gt_warps)
        time_dev = self._time_dev(w_t, w_v, gt_warps)

        self.log('train/time_err', time_err.mean())
        self.log('train/time_dev', time_dev.mean())
        self.log('train/time_err_epoch', time_err.mean(), on_epoch=True, on_step=False)
        self.log('train/time_dev_epoch', time_dev.mean(), on_epoch=True, on_step=False)
        if self.loss_fn == 'time_err':
            loss = time_err
        elif self.loss_fn == 'time_dev':
            loss = time_dev
        else:
            loss = dtw_cost
        self.log('train/loss', loss.mean())
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        i_b, s_t, p_t, s_f, p_f, l_g, u_g = batch
        s_f, p_f = self._extract_features(batch, batch_idx)
        gt_align = self.trainer.datamodule.val.gt
        with torch.no_grad():
            gt_warps = [self._ground_truth_warp(i, l, u, gt_align, s_t1, p_t1, p_f.device) for
                        i, l, u, s_t1, p_t1 in zip(i_b, l_g, u_g, s_t, p_t)]
        w_t, w_v, dtw_cost = self._predict_warp(s_f, p_f, s_t, p_t, gt_warps)

        time_err = self._time_err(w_t, w_v, gt_warps)
        time_dev = self._time_dev(w_t, w_v, gt_warps)

        self.log('val/time_err', time_err.mean())
        self.log('val/time_dev', time_dev.mean())
        if self.loss_fn == 'time_err':
            loss = time_err
        elif self.loss_fn == 'time_dev':
            loss = time_dev
        else:
            loss = dtw_cost
        self.log('val/loss', loss.mean())

    def test_step(self, batch, batch_idx):
        i_b, s_t, p_t, s_f, p_f, l_g, u_g = batch
        s_f, p_f = self._extract_features(batch, batch_idx)
        gt_align = self.trainer.datamodule.test.gt
        with torch.no_grad():
            gt_warps = [self._ground_truth_warp(i, l, u, gt_align, s_t1, p_t1, p_f.device) for
                        i, l, u, s_t1, p_t1 in zip(i_b, l_g, u_g, s_t, p_t)]

        w_t_cls, w_v_cls, dtw_cost_cls = self._predict_dtw(s_f, p_f, s_t, p_t)
        time_err_cls = self._time_err(w_t_cls, w_v_cls, gt_warps)
        time_dev_cls = self._time_dev(w_t_cls, w_v_cls, gt_warps)

        w_t_g, w_v_g, dtw_cost_g = self._predict_gdtw(s_f, p_f, s_t, p_t)
        time_err_g = self._time_err(w_t_g, w_v_g, gt_warps)
        time_dev_g = self._time_dev(w_t_g, w_v_g, gt_warps)

        metrics_full = {'test/time_err_dtw': time_err_cls,
                        'test/time_dev_dtw': time_dev_cls,
                        'test/time_err_gdtw': time_err_g,
                        'test/time_dev_gdtw': time_dev_g}
        self.log_dict({k: v.mean() for k, v in metrics_full.items()})
        return metrics_full

    def _extract_features(self, batch, batch_idx):
        i, s_t, p_t, s_f, p_f, l_g, u_g = batch
        if self.encoder == 'GRU':
            p_f1 = self.backbone_perf(p_f)[0]
            s_f1 = self.backbone_score(s_f)[0]
            p_f1 = nn.functional.normalize(p_f1, dim=2)
            s_f1 = nn.functional.normalize(s_f1, dim=2)
        else:
            p_f1 = self.backbone_perf(p_f)
            s_f1 = self.backbone_score(s_f)
        return s_f1, p_f1

    def _predict_warp(self, s_f, p_f, s_t, p_t, warp_gt):
        s_t1 = s_t - s_t[:, 0, None]
        p_t1 = p_t - p_t[:, 0, None]
        # aligns score to performance
        if type(self.dtw) is DecDTWLayer:
            return self._predict_gdtw(s_f, p_f, s_t1, p_t1)
        else:
            dtw_cost, w_t, w_v = self.dtw.forward(s_f, p_f, s_t1, p_t1, warp_gt)
            return w_t, w_v, dtw_cost

    def _predict_gdtw(self, s_f, p_f, s_t, p_t):
        ss = BatchedSignal(s_f, times=s_t)
        ps = BatchedSignal(p_f, times=p_t)
        warp_est = self.gdtw.forward(ps, ss, self.reg_wt)
        w_t, w_v = warp_est.times, warp_est.values
        dtw_cost = self.gdtw.dtw_objective(ps, ss, warp_est, self.reg_wt)
        return w_t, w_v, dtw_cost

    def _predict_dtw(self, s_f, p_f, s_t, p_t):
        dtw_cost, w_t, w_v = self.classicdtw.forward(s_f, p_f, s_t, p_t)
        w_t -= s_t[:, 0, None]
        w_v -= p_t[:, 0, None]
        return w_t, w_v, dtw_cost

    def _time_err(self, w_t, w_v, warp_gt):
        warp_pred = BatchedSignal(w_v, times=w_t)
        b, _ = warp_pred.shape
        loss = torch.zeros_like(warp_pred.times.max(dim=1).values)
        for i in range(b):
            gt_times, gt_values = warp_gt[i]
            S = gt_times[-1] - gt_times[0]
            ds = torch.diff(gt_times)
            warp_pred_at_gt = warp_pred[i](gt_times.unsqueeze(0))
            err = gt_values - warp_pred_at_gt.squeeze(0)
            e1, e2 = torch.abs(err[:-1]), torch.abs(err[1:])
            err1 = 0.5 * (e1 + e2)
            err2 = 0.5 * (e1 ** 2 + e2 ** 2) / (e1 + e2)
            sgn = torch.sign(err)
            sgn_eq = sgn[1:] == sgn[:-1]
            err1[~sgn_eq] = 0.
            err2[sgn_eq] = 0.
            dev = err1 + err2
            loss[i] = dev @ ds / S
        return loss

    def _time_dev(self, w_t, w_v, warp_gt):
        warp_pred = BatchedSignal(w_v, times=w_t)
        b, _ = warp_pred.shape
        loss = torch.zeros_like(warp_pred.times.max(dim=1).values)
        for i in range(b):
            gt_times, gt_values = warp_gt[i]
            S = gt_times[-1] - gt_times[0]
            ds = torch.diff(gt_times)
            warp_pred_at_gt = warp_pred[i](gt_times.unsqueeze(0))
            err = gt_values - warp_pred_at_gt.squeeze(0)
            e1, e2 = err[:-1], err[1:]
            se = 1 / 3. * (e1 ** 2 + e1 * e2 + e2 ** 2)
            loss[i] = torch.sqrt(se @ ds / S)
        return loss

    def _ground_truth_warp(self, i, lower, upper, gt_align, score_audio_t, perf_audio_t, device):
        gt_align_slc = gt_align[i][lower:upper].clone().to(device)
        gt_align_slc[:, 0] = gt_align_slc[:, 0] - score_audio_t[0]
        gt_align_slc[:, 1] = gt_align_slc[:, 1] - perf_audio_t[0]
        return gt_align_slc[:, 0], gt_align_slc[:, 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train score-to-audio alignment model')
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='base directory containing data')
    parser.add_argument('--slice_len', '-sl', type=int, default=256, help='length of slice of audio data for train/eval')
    parser.add_argument('--seed', '-sd', type=int, default=1, help='random seed to use for dataset generation/sampling')
    parser.add_argument('--feature_type', '-ft', type=str, required=True, choices=['cqt', 'chroma', 'melspec'],
                        help='base feature type used for alignment')
    parser.add_argument('--loss_fn', '-lf', type=str, required=True, choices=['time_err', 'time_dev', 'dtwnet', 'softdtw', 'l2atgt', 'dilate'],
                        help='loss function used for training')
    parser.add_argument('--encoder', type=str, choices=['GRU', 'None'], default='GRU', help='feature extractor backbone_perf type')
    parser.add_argument('--num_layers', '-nl', type=int, default=1, help='number of layers for GRU extractor')
    parser.add_argument('--out_dim', '-o', type=int, default=64, help='output feature dimensionality')
    parser.add_argument('--reg_wt', '-r', type=float, default=0.05, help='regularisation weight in DTW')
    parser.add_argument('--gamma', '-sg', type=float, default=1.0, help='soft-DTW gamma parameter')

    parser.add_argument('--num_epochs', '-n', type=int, default=20, help='number of epochs to run training')
    parser.add_argument('--n_gpus', '-g', type=int, default=1, help='number of gpus available for training')
    parser.add_argument('--batch_size', '-b', type=int, default=5, help='batch sized for training and eval')
    parser.add_argument('--lr', '-lr', type=float, default=0.0005, help='initial learning rate during training')
    parser.add_argument('--eval', '-e', action='store_true', help='evaluate model only')
    parser.add_argument('--ckpt', '-c', type=str, default='', help='checkpoint path (optional)')
    parser.add_argument("--wandb", action='store_true', help="enable wandb logging")

    args = parser.parse_args()

    dm = MusicDM(args.data_dir, args.feature_type, args.slice_len, args.batch_size, seed=args.seed)
    pl.seed_everything(args.seed)

    model = MusicDTWNet.load_from_checkpoint(args.ckpt) if args.ckpt else MusicDTWNet(**vars(args), n_warp_discr=args.slice_len * 2)

    if args.wandb:
        dtw_type = "gdtw" if args.loss_fn in ["time_err", "time_dev"] else "dtw"
        log_name = f'{args.feature_type}_{args.encoder}_{dtw_type}_slice:{args.slice_len}'
        if args.loss_fn in ['time_err', 'time_dev']:
            log_name += f'_reg:{args.reg_wt}'
        if args.encoder == 'GRU':
            log_name += f'_nlayers:{args.num_layers}_out:{args.out_dim}_{args.loss_fn}_lr:{args.lr}_bs:{args.batch_size}'
        if args.loss_fn == 'softdtw':
            log_name += f'_gamma:{args.gamma}'
        if args.loss_fn == 'dilate':
            log_name += f'_gamma:{args.gamma}'
        logger = WandbLogger(name=log_name, project='decldtw_audio')
    else:
        logger = False

    monitor = "val/loss" if args.loss_fn in ["time_err", "time_dev"] else "val/time_err"
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode="min", filename='{epoch:02d}-{val/loss:.4f}')
    trainer = pl.Trainer(max_epochs=args.num_epochs, devices=args.n_gpus, logger=logger, log_every_n_steps=10, #gradient_clip_val=1., #gradient_clip_algorithm='value',
                         val_check_interval=0.5, callbacks=[checkpoint_callback])

    if args.eval:
        trainer.validate(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path='best')
