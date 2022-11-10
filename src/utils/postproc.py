import numpy as np
import torch
from skimage.measure import label


import utils
from utils import log as log_utils

LOGGER = log_utils.getLogger(__name__)



def connected_filter(masks, filt=0.0, limit=None, dtype=None, device=None):
    """Extracts conneted components, sorts them by size and puts any extra into the largest (0th) segment"""
    dtype = dtype or masks.dtype
    device = device or masks.device
    hard_masks_cpu = utils.convert.to_5dim_hard_mask(masks, device='cpu', dtype=masks.dtype).squeeze(2)  # BxKxHxW
    k = int(limit or hard_masks_cpu.shape[1])
    segs = []
    rs = []
    for b in range(hard_masks_cpu.shape[0]):
        seg = label(hard_masks_cpu[b].argmax(0))
        lbls, counts = np.unique(seg, return_counts=True)
        reindex = (-counts).argsort()
        new_lbls = lbls[reindex]

        mask = counts < -1
        # Filter out any masks that are too small
        if filt > 0:
            mask = counts < np.prod(seg.shape[-2:]) * filt
            mask = mask[reindex]  # put into decreasing size order

        if k > 0:
            mask[k:] = True

        for new_label, (lbl, set_zero) in enumerate(zip(new_lbls, mask), start=1):
            nl = -1 if set_zero else -new_label
            seg = np.where(seg == lbl, nl, seg)

        seg = -seg - 1
        segs.append(seg)
        rs.append(np.sum(~mask))

    segs = torch.from_numpy(np.array(segs))
    K = np.max(rs)
    return utils.convert.to_5dim_hard_mask(segs, k, device=device, dtype=dtype)
