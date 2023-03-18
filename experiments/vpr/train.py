import argparse
import os, sys
from os.path import join
import pickle

import sklearn
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from google_drive_downloader import GoogleDriveDownloader as gdd
import wandb

sys.path.append('../../')
from decdtw.utils import BatchedSignal, dtw_objective
from decdtw.decdtw import DecDTWLayer
from data import DTWDatamodule

sys.path.append("./thirdparty/deep-visual-geo-localization-benchmark/")
from model import network


from comparisons import SoftDTW, DILATE, DTWLayer
sdtw_layer = SoftDTW(gamma=1, subseq=True)
dilate_layer = DILATE(subseq=True, alpha=0., gamma=0.1)

# torch.use_deterministic_algorithms(True)
pl.seed_everything(1)

OFF_THE_SHELF_RADENOVIC = {
    'resnet50conv5_sfm'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
}

OFF_THE_SHELF_NAVER = {
    "resnet50conv5"  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}


def pca_get_data(pca):
    return torch.FloatTensor(pca.mean_).cuda(), torch.FloatTensor(pca.components_).cuda()


def pca_transform(features,m,c):
    return torch.matmul(features - m , torch.t(c))


def load_weights(args, model):
    if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
        if args.off_the_shelf.startswith("radenovic"):
            pretrain_dataset_name = args.off_the_shelf.split("_")[1]  # sfm or gldv1 datasets
            url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
            state_dict = load_url(url, model_dir=join("data", "off_the_shelf_nets"))
        else:
            # This is a hacky workaround to maintain compatibility
            sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca
            zip_file_path = join("data", "off_the_shelf_nets", args.backbone + "_naver.zip")
            if not os.path.exists(zip_file_path):
                gdd.download_file_from_google_drive(file_id=OFF_THE_SHELF_NAVER[args.backbone],
                                                    dest_path=zip_file_path, unzip=True)
            if args.backbone == "resnet50conv5":
                state_dict_filename = "Resnet50-AP-GeM.pt"
            elif args.backbone == "resnet101conv5":
                state_dict_filename = "Resnet-101-AP-GeM.pt"
            state_dict = torch.load(join("data", "off_the_shelf_nets", state_dict_filename))
        state_dict = state_dict["state_dict"]
        model_keys = model.state_dict().keys()
        renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
        model.load_state_dict(renamed_state_dict)
    elif args.resume != None:
        print(f'loading pretrained model {args.resume}')
        state_dict = torch.load(args.resume)["model_state_dict"]
        model.load_state_dict(state_dict)
    else:
        print(f"Using off-the-shelf:{args.off_the_shelf} model pretrained on {args.pretrain} for evaluation")

    return model


def huber_loss(x, delta):
    return F.huber_loss(x, torch.zeros_like(x), reduction='none', delta=delta)


def ate_sq_loss(x, delta=None):
    return x ** 2.


def id_loss(x, delta=None):
    return x


def trunc_quad_loss(x, delta):
    return ate_sq_loss(torch.clamp(x, max=delta))


class DTWVPR(pl.LightningModule):
    def __init__(self, args=dict(), lr=1e-5, reg_wt=0.1, scheduler_params={'step': 1, 'gamma': 1.0}, dtw_discr=50, ckpt_path='',
                 backbone='vgg16', aggregation='netvlad', netvlad_clusters=64, criterion_fn='square', criterion_agg='mean',
                 criterion_delta=5., learnable_params_kwds=['aggregation'], eval_thres=[1, 2, 3, 4, 5, 10, 15, 20], gru=False):
        super().__init__()
        # feature extractor params
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.agg_layer = aggregation
        self.pca = False
        if self.agg_layer == 'netvlad':
            self.netvlad_clusters = args.netvlad_clusters
        self.model = network.GeoLocalizationNet(args)
        if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
            args.features_dim *= args.netvlad_clusters
        self.features_dim = args.features_dim
        if args.pca_dim is not None:
            self.pca = True
            pca_model_filename = f'{args.resume}_pca_{args.pca_dim}.pkl'
            pca_model = pickle.load(open(pca_model_filename,'rb'))
            self.p_m, self.p_c = pca_get_data(pca_model)
            self.features_dim = args.pca_dim
        # the following line needed because of https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
        self.model = torch.nn.DataParallel(self.model)
        self.model = load_weights(args, self.model)
        # freeze weights for training based on keywords
        for name, param in self.model.named_parameters():
            for kwd in learnable_params_kwds:
                if kwd in name:
                    param.requires_grad = True
                    break
            else:
                param.requires_grad = False
        self.learnable_params_kwds = learnable_params_kwds
        self.gru = gru
        if gru:
            self.gru_layer = nn.GRU(self.features_dim, self.features_dim, batch_first=True, num_layers=1)
        self.lr = lr
        self.scheduler_params = scheduler_params
        self.reg_wt = reg_wt
        self.criterion_fn, self.criterion_agg = criterion_fn, criterion_agg
        self.criterion_delta = criterion_delta
        if criterion_fn in ["softdtw", "dilate", "alonggt"]:
            self.dtw = DTWLayer(subseq=True)
        else:
            self.dtw = DecDTWLayer(subseq_enabled=True, n_warp_discr=dtw_discr)
        self.eval_thres = eval_thres
        self.save_hyperparameters(ignore=['eval_thres'])

    def compute_pose_loss(self, batch, batch_idx):
        imgs_query = batch['imgs_query']
        imgs_db = batch['imgs_db']
        ts_query = batch['ts_query']
        ts_db = batch['ts_db']
        ts_min_err = batch['ts_min_err']
        xy_query = batch['xy_query']
        xy_db = batch['xy_db']
        B, Nq = ts_query.shape
        B, Nd = ts_db.shape
        C, H, W = imgs_query.shape[2:]

        # compute time warp registration using images
        descr_query = self.model(imgs_query.reshape(-1, C, H, W)).reshape(B, Nq, -1)
        descr_db = self.model(imgs_db.reshape(-1, C, H, W)).reshape(B, Nd, -1)
        if self.pca:
            descr_db = pca_transform(descr_db, self.p_m, self.p_c)
            descr_query = pca_transform(descr_query, self.p_m, self.p_c)
        if self.gru:
            descr_query = self.gru_layer(descr_query, descr_query[None, :, 0, :].clone())[0]
            descr_db = self.gru_layer(descr_db, descr_db[None, :, 0, :].clone())[0]
        signal_descr_query = BatchedSignal(descr_query, times=ts_query)
        signal_descr_db = BatchedSignal(descr_db, times=ts_db)
        opt_warp_gps = BatchedSignal(ts_min_err, times=ts_query)
        if self.criterion_fn == 'softdtw':
            dtw_loss = sdtw_layer(descr_db, descr_query)
            _, w_ts, w_vs = self.dtw(descr_query, descr_db, ts_query, ts_db)
            est_warp_path = BatchedSignal(w_vs, times=w_ts)
        elif self.criterion_fn == 'dilate':
            dtw_loss = dilate_layer(descr_query, descr_db, xy_query, xy_db)
            _, w_ts, w_vs = self.dtw(descr_query, descr_db, ts_query, ts_db)
            est_warp_path = BatchedSignal(w_vs, times=w_ts)
        elif self.criterion_fn == 'alonggt':
            dtw_loss = dtw_objective(signal_descr_db, signal_descr_query, opt_warp_gps, self.reg_wt)
            _, w_ts, w_vs = self.dtw(descr_query, descr_db, ts_query, ts_db)
            est_warp_path = BatchedSignal(w_vs, times=w_ts)
        elif self.criterion_fn == 'single':
            dtw_loss = torch.randn((B,), dtype=descr_db.dtype, device=descr_db.device)  # placeholder
        else:
            est_warp_path = self.dtw.forward(signal_descr_db, signal_descr_query, self.reg_wt)
            dtw_loss = self.dtw.dtw_objective(signal_descr_db, signal_descr_query, est_warp_path, self.reg_wt)

        # project time warp to GPS error (m)
        signal_gps_db = BatchedSignal(xy_db, times=ts_db)
        if self.criterion_fn == 'single':
            sims = torch.einsum('bmd,bnd->bmn', descr_query, descr_db)
            best_match_db_ind = sims.argmax(dim=2, keepdim=True)
            xy_pred = torch.gather(xy_db, 1, best_match_db_ind.repeat(1, 1, 2)) 
        else:
            xy_pred = signal_gps_db(est_warp_path.values)
        xy_err_per_query = (xy_query - xy_pred).norm(dim=2)

        return xy_err_per_query, dtw_loss

    def log_losses(self, xy_err_per_query, dtw_loss, mode='train', traverse=''):
        ate = ate_sq_loss(xy_err_per_query).mean(dim=1) ** 0.5
        max_err = xy_err_per_query.max(dim=1).values
        B = ate.shape[0]
        if mode in ['train', 'val']:
            if self.criterion_fn == 'square':
                train_loss_fn = ate_sq_loss
            elif self.criterion_fn == 'huber':
                train_loss_fn = huber_loss
            elif self.criterion_fn == 'trunc':
                train_loss_fn = trunc_quad_loss
            elif self.criterion_fn == 'single':
                train_loss_fn = id_loss
            else:
                train_loss_fn = None
            if train_loss_fn is not None:
                train_loss = train_loss_fn(xy_err_per_query, self.criterion_delta)  # per query loss
                train_loss = train_loss.mean(dim=1) if self.criterion_agg == 'mean' else train_loss.max(dim=1).values
                train_loss = train_loss.mean()  # aggregate over batch
            else:
                train_loss = dtw_loss.mean()  # softdtw and alonggt training
            self.log(f'{mode}/loss/criterion', train_loss, batch_size=B) 
        else:
            train_loss = None
        # log losses
        if mode != 'test':
            self.log(f'{mode}/loss/ate', ate.mean(), batch_size=B)
            self.log(f'{mode}/loss/max_err', max_err.mean(), batch_size=B)
        else:
            self.log(f'{mode}/{traverse}/loss/ate', ate.mean(), batch_size=B, add_dataloader_idx=False)
            self.log(f'{mode}/{traverse}/loss/max_err', max_err.mean(), batch_size=B, add_dataloader_idx=False)
        # log accuracies
        for thres in self.eval_thres:
            if mode != 'test':
                self.log(f'{mode}/accuracy/ate_{str(thres).zfill(2)}m', (ate < thres).float().mean(), batch_size=B)
                self.log(f'{mode}/accuracy/max_{str(thres).zfill(2)}m', (max_err < thres).float().mean(), batch_size=B)
            else:
                self.log(f'{mode}/{traverse}/accuracy/ate_{str(thres).zfill(2)}m', (ate < thres).float().mean(),
                         batch_size=B, add_dataloader_idx=False)
                self.log(f'{mode}/{traverse}/accuracy/max_{str(thres).zfill(2)}m', (max_err < thres).float().mean(),
                         batch_size=B, add_dataloader_idx=False)
        return train_loss

    def training_step(self, batch, batch_idx):
        xy_err, dtw_loss = self.compute_pose_loss(batch, batch_idx)
        loss = self.log_losses(xy_err, dtw_loss, mode='train')
        return loss

    def validation_step(self,  batch, batch_idx):
        xy_err, dtw_loss = self.compute_pose_loss(batch, batch_idx)
        self.log_losses(xy_err, dtw_loss, mode='val')
        return None

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        xy_err, dtw_loss = self.compute_pose_loss(batch, batch_idx)
        traverse = self.trainer.datamodule.test_seqs[dataloader_idx].dataset_file.split('_')[-1][:-4]
        self.log_losses(xy_err, dtw_loss, mode='test', traverse=traverse)
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler_params is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.scheduler_params['step'], gamma=self.scheduler_params['gamma'])
            return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
        else:
            return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--eval', '-e', action='store_true', help='evaluate model only')
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Number of paired sequences in a batch.")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validation and testing)")
    parser.add_argument("--criterion", type=str, default='square,mean', help="loss to be used in form 'per_query_loss,aggregation'",
                        choices=["square,mean", "square,max", "huber,mean", "huber,max", "trunc,mean",
                                 "trunc,max", "softdtw", "alonggt", "single", "dilate"])
    parser.add_argument("--criterion-delta", type=float, default=5., help="criterion delta for huber/truncated quadratic loss")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--reg-wt", type=float, default=0.1, help="regularisation weight for dtw layer")
    parser.add_argument("--gru", action='store_true', help="add a gru layer on top of feature extractor")
    parser.add_argument("--scheduler-step", type=int, default=1, help="step for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=1.0, help="gamma for StepLR scheduler")
    parser.add_argument("--learnable-params", type=str, nargs='+', default=['aggregation', '28'], help='keywords for identifying learnable parameters')
    parser.add_argument("--val_check_interval", type=float, default=0.5, help='how often to run validation epoch as fraction of train epoch')
    parser.add_argument("--eval-thres", type=int, nargs='+', default=[1, 2, 3, 4, 5, 10, 15, 20], help='evaluation threshold in meters')
    # Logging parameters
    parser.add_argument("--wandb", action='store_true', help="enable wandb logging")
    parser.add_argument("--offline", action='store_true', help="generate wandb logs offline")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5", 
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="netvlad", choices=["netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm",
                                                                               "cls", "seqpool", "none"])
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--num_non_local', type=int, default=1, help="Num of non local blocks")
    parser.add_argument("--non_local", action='store_true', help="_")
    parser.add_argument('--channel_bottleneck', type=int, default=128, help="Channel bottleneck for Non-Local blocks")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    # Initialization parameters
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--img-path", type=str, required=True, help="Robotcar base image path")
    parser.add_argument("--seq-list-path", type=str, required=True, help="path containing paired sequence lists and ground truth")
    # Other parameters

    parser.add_argument("--num_workers", type=int, default=16, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--accum-batches", type=int, default=1, help="number of batches to accumulate for gradients")
    # Paths parameters
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument('--ckpt', '-c', type=str, default='', help='lightning checkpoint path (optional)')
    parser.add_argument('--n-gpu', '-g', type=int, default=1, help='number of gpus to use')

    args = parser.parse_args()
    
    if args.aggregation == "crn" and args.resume == None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")

    dm = DTWDatamodule(img_base_dir=args.img_path, data_list_dir=args.seq_list_path, batch_size_train=args.train_batch_size,
                       batch_size_eval=args.infer_batch_size, num_workers=args.num_workers)
    scheduler_params = None if args.scheduler_gamma == 1.0 else {'step': args.scheduler_step, 'gamma': args.scheduler_gamma}

    if args.criterion not in ['softdtw', 'alonggt', 'single', 'dilate']:
        crit_fn, crit_agg = args.criterion.split(',')
    else:
        crit_fn, crit_agg = args.criterion, 'mean'

    val_loss_ckpt = ModelCheckpoint(monitor='val/loss/ate', mode='min', save_on_train_epoch_end=False)

    if args.wandb:
        name = f'{args.backbone}_{args.aggregation}_reg+{args.reg_wt}'
        if args.eval:
            name += f'_eval'
            if args.ckpt:
                name += f'+{args.ckpt}'
            if crit_fn == 'single':
                name += '_single'
        else:
            name +=  f'_{"+".join(args.learnable_params)}_lr+{args.lr}_{args.criterion}'
            if crit_fn in ['huber', 'trunc']:
                name += f'+{args.criterion_delta}'
        logger = WandbLogger(name=name, project='vpr_dtw', group='dtw_finetune', offline=args.offline)
        wandb.run.name += '_' + wandb.run.id
        wandb.run.save()
    else:
        logger = False

    vpr = DTWVPR(args=args, lr=args.lr, reg_wt=args.reg_wt, ckpt_path=args.resume, backbone=args.backbone, aggregation=args.aggregation,
                 netvlad_clusters=args.netvlad_clusters, criterion_fn=crit_fn, criterion_agg=crit_agg, criterion_delta=args.criterion_delta,
                 scheduler_params=scheduler_params, learnable_params_kwds=args.learnable_params,
                 eval_thres=args.eval_thres, gru=args.gru)
    if args.ckpt:
        vpr = DTWVPR.load_from_checkpoint(args.ckpt)
        vpr.criterion_fn = crit_fn

    trainer = pl.Trainer(devices=args.n_gpu, val_check_interval=args.val_check_interval, max_epochs=args.epochs_num, logger=logger,
                         log_every_n_steps=81, reload_dataloaders_every_n_epochs=1,
                         accumulate_grad_batches=args.accum_batches, callbacks=[val_loss_ckpt])
    # evaluate before and after training for comparison
    if args.eval or crit_fn == 'single':
        trainer.validate(vpr, dm)
        trainer.test(vpr, dm)
    else:
        trainer.validate(vpr, dm)
        trainer.fit(vpr, dm)
        trainer.test(vpr, dm, ckpt_path='best')

