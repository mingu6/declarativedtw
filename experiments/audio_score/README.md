# Audio data experiment setup

# Dataset setup

The separately provided archive `alignment-eval.zip` in the supplemetary materials attached to the paper contains code to generate the dataset required for the audio experiments.

# Run training/eval of models

Use `train.py` to train and evaluate models. See below for usage

```
usage: train.py [-h] --data_dir DATA_DIR [--slice_len SLICE_LEN] --feature_type {cqt,chroma,melspec} --loss_fn {time_err,time_dev,dtwnet,softdtw,l2atgt} [--encoder {GRU,None}]
                [--num_layers NUM_LAYERS] [--out_dim OUT_DIM] [--reg_wt REG_WT] [--gamma GAMMA] [--num_epochs NUM_EPOCHS] [--n_gpus N_GPUS] [--batch_size BATCH_SIZE] [--lr LR]
                [--eval] [--ckpt CKPT]

train score-to-audio alignment model

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -d DATA_DIR
                        base directory containing data
  --slice_len SLICE_LEN, -sl SLICE_LEN
                        length of slice of audio data for train/eval
  --feature_type {cqt,chroma,melspec}, -ft {cqt,chroma,melspec}
                        base feature type used for alignment
  --loss_fn {time_err,time_dev,dtwnet,softdtw,l2atgt}, -lf {time_err,time_dev,dtwnet,softdtw,l2atgt}
                        loss function used for training
  --encoder {GRU,None}  feature extractor backbone type
  --num_layers NUM_LAYERS, -nl NUM_LAYERS
                        number of layers for GRU extractor
  --out_dim OUT_DIM, -o OUT_DIM
                        output feature dimensionality
  --reg_wt REG_WT, -r REG_WT
                        regularisation weight in DTW
  --gamma GAMMA, -sg GAMMA
                        soft-DTW gamma parameter
  --num_epochs NUM_EPOCHS, -n NUM_EPOCHS
                        number of epochs to run training
  --n_gpus N_GPUS, -g N_GPUS
                        number of gpus available for training
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch sized for training and eval
  --lr LR, -lr LR       initial learning rate during training
  --eval, -e            evaluate model only
  --ckpt CKPT, -c CKPT  checkpoint path (optional)
```

For example, to train our system on constant-Q transform features run `python3 train.py -d /path/to/alignment-eval/ -ft cqt -lf time_err --encoder GRU -lr 0.0005 -nl 1 -o 128 -r 0.08 -n 20`.
