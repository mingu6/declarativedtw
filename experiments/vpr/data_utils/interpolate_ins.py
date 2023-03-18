import argparse
from collections import defaultdict
import csv
import os
import math
import sys

import numpy as np
import pandas as pd


bad_traverses = ['2014-05-19-13-20-57', '2014-05-19-13-05-38', '2015-08-21-10-40-24']  # very dodgy INS for these traverses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oxford-sdk', type=str, required=True, help='path to RobotCar dataset sdk repo')
    parser.add_argument('--image-list', type=str, required=True, help='txt file with image filenames from Thoma et al.')
    parser.add_argument('--data-dir', type=str, required=True, help='base directory of raw data')
    parser.add_argument('--output-dir', type=str, required=True, help='where to store pose data')
    args = parser.parse_args()

    sys.path.append(args.oxford_sdk)
    from robotcar_dataset_sdk.python.interpolate_poses import interpolate_ins_poses
    from robotcar_dataset_sdk.python.transform import se3_to_components

    # partition image timestamps from fnames by traverse - process ins interpolations individually per traverse
    ts_split = defaultdict(list)
    with open(args.image_list, 'r') as f:
        reader = csv.reader(f, delimiter='/')
        for stuff, fname in reader:
            traverse = stuff.split('_')[0]
            if traverse in bad_traverses:
                continue
            ts = int(fname[:-4])
            ts_split[traverse].append(ts)

    data = pd.DataFrame([], columns=["traverse", "fname", "northing", "easting", "yaw"])

    # interpolate poses, save only x, y, yaw
    for traverse, tstamps in ts_split.items():
        try:
            rtk_path = os.path.join(args.data_dir, 'rtk', traverse, 'rtk.csv')
            ins_path = os.path.join(args.data_dir, 'images', traverse, 'gps', 'ins.csv')
            rtk_exists_flag = os.path.isfile(rtk_path)
            if rtk_exists_flag:
                poses_se3 = interpolate_ins_poses(rtk_path, tstamps, use_rtk=True)
            else:
                poses_se3 = interpolate_ins_poses(ins_path, tstamps)
            poses = [se3_to_components(mat)[[0, 1, 5]] for mat in poses_se3]
            if rtk_exists_flag:
                poses = np.array(poses)
                poses[:, 2] += 3 * math.pi / 2  # RTK and INS yaw off by 3pi/2 radians for some reason??
                poses[:, 2] = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))  # ensures angles b/w [-pi, pi]
            df = pd.DataFrame(poses, columns=["northing", "easting", "yaw"])
            fnames = [str(ts) + '.png' for ts in tstamps]
            traverses = [traverse] * len(fnames)
            df.insert(0, "fname", fnames, True)
            df.insert(0, "traverse", traverses, True)
            data = data.append(df)
        except Exception as e:
            print(str(e))

    split = os.path.basename(args.image_list)[:-4]
    data.to_csv(os.path.join(args.output_dir, split + ".csv"), index=False)
