import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from cvbase.optflow.visualize import flow2rgb
from detectron2.data import detection_utils as d2_utils

__LOGGER = logging.Logger(__name__)
__TAR_SP = [Path('/usr/bin/tar'), Path('/bin/tar')]

TAG_FLOAT = 202021.25


@lru_cache(None)
def __tarbin():
    for p in __TAR_SP:
        if p.exists():
            return str(p)
    __LOGGER.error(f"Could not locate tar binary")
    return 'tar'


def tar(*args):
    arg_list = [__tarbin(), *args]
    __LOGGER.info(f"Executing {arg_list}")
    print(f"Executing {arg_list}")
    return subprocess.check_call(arg_list, close_fds=True)


def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

class UnindetifedFlowError(ValueError):
    pass

def read_flo2(file):
    if not os.path.isfile(file):
        raise UnindetifedFlowError("file does not exist %r" % str(file))
    if str(file)[-4:] != '.flo':
        raise UnindetifedFlowError("file ending is not .flo %r" % file[-4:])
    try:
        with open(file, 'rb') as f:
            flo_number = np.fromfile(f, np.float32, count=1)[0]
            if flo_number != TAG_FLOAT:
                raise UnindetifedFlowError('Flow number %r incorrect. Invalid .flo file' % flo_number)
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (int(h), int(w), 2))

    # What about numpy error?
    except (EOFError, IOError, OSError) as e:
        raise UnindetifedFlowError(str(e))
    return flow


def read_flow(sample_dir, resolution=None, to_rgb=False, ccrop=False, crop_frac=1.0):
    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)
    if ccrop:
        s = int(min(h, w) * crop_frac)
        flow = flow[(h-s) // 2: (h-s) // 2 + s, (w-s) // 2: (w-s) // 2 + s]
        h = s
        w = s
    if resolution:
        flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
        flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
        flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    if to_rgb:
        flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return einops.rearrange(flow, 'h w c -> c h w')

def read_flow2(sample_dir, resolution=None, to_rgb=False, ccrop=False, crop_frac=1.0):
    flow = read_flo2(sample_dir)
    h, w, _ = np.shape(flow)
    if ccrop:
        s = int(min(h, w) * crop_frac)
        flow = flow[(h-s) // 2: (h-s) // 2 + s, (w-s) // 2: (w-s) // 2 + s]
        h = s
        w = s
    if resolution:
        flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
        flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
        flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    if to_rgb:
        flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return einops.rearrange(flow, 'h w c -> c h w')
