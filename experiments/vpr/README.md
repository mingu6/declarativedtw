# Visual place recognition (VPR) experiments

## Dataset setup

### Directory structure

From the base directory the RobotCar dataset should be structured as follows. We will describe in subsequent sections how to download the raw data
and preprocess it to be ready for training.

```
raw_data
├─ images
|  ├─ 201x-xx-xx-xx-xx-xx
|  |  ├─ gps
|  |  |  ├─ gps.csv
|  |  |  └─ ins.csv
|  |  ├─ stereo
|  |  |  └─ centre
|  |  |     ├─ 1400505xxxxxxxxx.png
|  |  |     ├─ ...
|  ├─ ...
└─ rtk
   ├─ 201x-xx-xx-xx-xx-xx
   |  └─ rtk.csv
   ├─ ...
ready_data
├─ images
|  ├─ 201x-xx-xx-xx-xx-xx
|  |  ├─ stereo
|  |  |  └─ centre
|  |  |     ├─ 1400505xxxxxxxxx.png
|  |  |     ├─ ...
seq_refine_splits
├─ sequence_pairs_train.csv
├─ ...
single_image_splits
├─ train_database.csv
├─ ...
thoma_lists
├─ test
|  ├─ oxford_night_query.txt
|  ├─ ...
├─ train
|  ├─ train.txt
|  └─ validation_query.txt
```

### Download raw images and GPS data

Use the [RobotCar dataset scraper](https://github.com/mttgdd/RobotCarDataset-Scraper) to download the raw image and GPS + INS data required for training.
For convenience, we have included dataset lists which can be run with the scraper in the `vpr.zip` archive under the `scraper_lists` folder in the supplementary materials associated with the paper. Download
all data into the `raw_data/images` folder in the base directory structure.

Also, download the [RTK ground truth](https://robotcar-dataset.robots.ox.ac.uk/ground_truth/) data into `raw_data` folder under the base directory.

### Preprocess images

Next, use `data_utils/preprocess_images.py` to debayerise and undistort the raw images. Use the dataset lists in the `thoma_lists` folder provided in
`vpr.zip`. We only apply preprocessing on the selected images to conserve compute time and hard disk space. You will need the [Robotcar dataset sdk](https://github.com/ori-mrg/robotcar-dataset-sdk) for this step, note the separate dependencies (would recommend separate virtualenv).

## Train single image baseline

We use a modified version of the [deep visual geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) provided
in `dvg.zip` to train our single image VPR baseline, used as the starting point for sequence fine-tuning. Unzip this archive into the `thirdparty`
folder under the base directory. We modified the dataset format and the positive mining methodology which we found yielded better results on Oxford.
Specifically, for positive mining, we take 10 random images within 10 meters to an anchor rather than necessarily the 10 nearest images.

The train/validation/test image lists are provided in the `single_image_splits` folder in `vpr.zip`. You will need to refer to this folder in the
arguments of the training script for the code to understand train/val/test (and query/database) splits. Use the below command to train the single
image method after setting up the dependencies.

```
python3 thirdparty/deep-visual-geo-localization-benchmark/train.py --dataset_name oxford --datasets_folder /path/to/datasets_vg/datasets/ --expName posMine-rand10_pool-NV_back-vgg16 --aggregation netvlad --loadDatasetFromFile --backbone vgg16 --resize 180 240 --pos_mine_method random_10 --image_root_dir /path/to/base/data/dir/ready_data/images/ --csv_dir /path/to/base/data/dir/single_image_splits/
```

You will need to install the dependencies for the deep visual geo-localization benchmark project (use their `requirements.txt`) before running our sequence fine-tuning.

## Train using sequence maximum GPS error loss

Use `train.py` to train the VPR network. Once the single image baseline has been trained, we can use the best model checkpoint to initialise 
sequence-based fine-tuning using declarative DTW. Use the following command as an example to train the NetVLAD aggregation and Conv5 layers. As is,
this requires a lot of GPU memory (Nvidia RTX 3090 24Gb should be enough) because NetVLAD descriptors are 32k dim, however you can reduce the batch size and accumulate training batches using the `--accum-batches num_batches` command to reduce memory requirements.

```
python3 train.py --aggregation netvlad --backbone vgg16 --resize 180 240 --resume /path/to/single/image/checkpoint/best_model.pth --criterion square,max --learnable-params aggregation 28 --epochs_num 30 --lr 0.0001 --val_check_interval 0.2 --train_batch_size 8 --wandb 
```

TODO: cache descriptors up to first learnable layer and train over the cache.
