import os
import numpy as np
from Geometric import G_config as conf

def load_geometric_data():
    data_path = conf.data_path

    train_dat = np.load(os.path.join(data_path, "shuf_trn_dat.npy"), mmap_mode="r")
    valid_dat = np.load(os.path.join(data_path, "shuf_vld_dat.npy"), mmap_mode="r")
    red_test_dat = np.load(os.path.join(data_path, "trn_vld_tst/red_tst_dat.npy"), mmap_mode="r")
    orange_test_dat = np.load(os.path.join(data_path, "trn_vld_tst/orange_tst_dat.npy"), mmap_mode="r")
    train_lbl = np.load(os.path.join(data_path, "shuf_trn_lbl.npy"), mmap_mode="r")
    valid_lbl = np.load(os.path.join(data_path, "shuf_vld_lbl.npy"), mmap_mode="r")

    return train_dat, train_lbl, valid_dat, valid_lbl, red_test_dat, orange_test_dat

def code_creator(size, MINI=True):
    if MINI == True:
        for i in range(0, size):
            code = np.random.choice(2, 2, replace=False).astype("float32")
            code = np.expand_dims(code, axis=0)
            if i == 0: target_c = code
            else: target_c = np.append(target_c, code, axis=0)
    return target_c

def codemap(target_c):
    one = np.argmax(target_c, axis=-1)
    c1, c2, c3 = np.zeros((len(target_c), 32, 32, 2)), np.zeros((len(target_c), 16, 16, 2)), np.zeros((len(target_c), 8, 8, 2))
    for i in range(len(target_c)):
        c1[i, :, :, one[i]], c2[i, :, :, one[i]], c3[i, :, :, one[i]] = 1., 1., 1.
    return c1, c2, c3

def flip(x):
    flip = x - 1
    return np.where(flip == -1, 1, flip)

def label_smoothing(labels, factor):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels