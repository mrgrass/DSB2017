import numpy as np
import dicom
import glob
import os
import cv2
import mxnet as mx
import pandas as pd
from tqdm import tqdm

def get_extractor():
    model = mx.model.FeedForward.load('resnet/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)
    return feature_extractor

def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])

def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch

def calc_features():
    net = get_extractor()
    for folder in tqdm(glob.glob('../input/stage1/*')):
    # for folder in glob.glob('../input/stage1/*'):
        batch = get_data_id(folder)
        feats = net.predict(batch)
        np.save(folder, feats)

if __name__ == '__main__':
    print("Starting ...")
    calc_features()
