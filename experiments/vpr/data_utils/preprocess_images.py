import argparse
import csv
from multiprocessing import Pool
import os
import sys

import numpy as np
import cv2
import time


# taken from Janine's repository utils folder
def resize_img(img, max_size):
    scale = max_size / float(max(img.shape[0], img.shape[1]))
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)


def save_img(img, out_file):
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_file), img)


def func(stuff, fname):
    traverse = stuff.split('_')[0]
    img_fpath = os.path.join(args.data_dir, traverse, 'stereo', 'centre', fname)
    img_outpath = os.path.join(args.output_dir, traverse, 'stereo', 'centre', fname)
    if os.path.isfile(img_outpath):
        return None
    elif os.path.isfile(img_fpath):
        os.makedirs(os.path.dirname(img_outpath), exist_ok=True)
        try:
            img = image.load_image(img_fpath, model=cm)
            img = resize_img(img, 240)          
            save_img(img, img_outpath)
            print('processed ' + img_outpath)
        except:
            print('error processing ' + img_outpath)
    else:
        print(img_fpath + ' is missing!')
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oxford-sdk', type=str, required=True, help='path to RobotCar dataset sdk repo')
    parser.add_argument('--image-list', type=str, required=True, help='txt file with image filenames from Thoma et al.')
    parser.add_argument('--data-dir', type=str, required=True, help='base directory of raw images')
    parser.add_argument('--output-dir', type=str, required=True, help='where to store pose images')
    args = parser.parse_args()

    sys.path.append(args.oxford_sdk)
    import image
    from camera_model import CameraModel
    camera_models_dir = os.path.join(os.path.abspath(os.path.join(args.oxford_sdk, '..')), "models")

    dummy_img_path = os.path.join('stereo', 'centre')
    cm = CameraModel(camera_models_dir, dummy_img_path)

    with open(args.image_list, 'r') as f:
        reader = csv.reader(f, delimiter='/')
        for stuff, fname in reader:
            func(stuff, fname)
        # pool = Pool(processes=16)   # use instead of prev two lines for multiprocessing
        # pool.starmap(func, reader)


