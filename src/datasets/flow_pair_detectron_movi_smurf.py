import itertools
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fvt
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from torch.utils.data import Dataset

from datasets.clevrmov import FlatCLEVRMOV
from utils.data import read_flow

logger = logging.getLogger('unsup_vidseg')


class FlowPairMoviDetectronSmurf(Dataset):
    type_2_suffix = {
        'rgb': '.jpg',
        'ano': '.png',
        'fwd': '.npz',
        'bwd': '.npz'
    }


    def __init__(self, split, flow_dir, resolution, prefix=None, cache_path=None, two_flow=False, gt_flow=False, flow_clip=float('inf'), norm=False, num_frames=1):
        self.ignore_label = -1
        prefix = prefix.split('.')[0]
        self.resolution = resolution
        print(self.resolution)
        self.gt_flow = gt_flow
        self.two_flow = two_flow
        self.random = split == 'train'
        self.split = split
        self.to_rgb = False
        self.dataset_key = Path(prefix).name
        self.cache_path = cache_path
        self.num_frames = num_frames
        print(f"Assuming dataset name is {self.dataset_key}")
        self.path = Path(prefix) / str(split)
        self.index = []
        self.inv_index = {}
        for f in self.path.glob(f"{self.dataset_key}_*"):
            if f.is_dir():
                for fid in range(0 if self.gt_flow else 1, 24 if self.gt_flow else 22):
                    if (f / self.make_key(f.name, fid)).exists():
                        self.index.append((f, f.name, fid))
                        self.inv_index[(f.name, fid)] = len(self.index) - 1
                    else:
                        logger.warning(f"Missing {f.name} frame {fid}")

        # self.dataset.tpstr_to_index_map = {str(tp.name): ind for (tp, ind) in self.dataset.inds.values()}
        # self.dataset.bias = 0
        # self.dataset.limit = len(self.dataset.inds)
        self.flow_dir = flow_dir or f'data/smurf/flow_out/{self.dataset_key}/{"train" if split == "train" else "validation" }/' #or str(Path(prefix) / 'raft_new')
        self.data_dir = [flow_dir, flow_dir, flow_dir]

        logger.info(f"{self.dataset_key.upper()} {self.split}: Prefix={self.path}")
        self.flow_clip = flow_clip
        self.norm_flow = norm

        if self.num_frames is None or self.num_frames > 1:
            self.seq_fid_2_idx = defaultdict(lambda: defaultdict(int))
            for idx in range(len(self.index)):
                prefix, fkey, fid = self.index[idx]
                self.seq_fid_2_idx[fkey][fid] = idx
            self.seq_index = []
            for seq in self.seq_fid_2_idx:
                min_fid = min(self.seq_fid_2_idx[seq])
                max_fid = max(self.seq_fid_2_idx[seq])
                if self.num_frames is None:
                    self.seq_index.append((seq, min_fid))
                else:
                    for fi in range(min_fid, max_fid + 1 - self.num_frames, 1 if self.random else self.num_frames):
                        self.seq_index.append((seq, fi))
        pairs = [1, 2, -1, -2]
        self.tp_to_flow_prefix_map = {}
        if not self.gt_flow:
            for f in self.path.glob(f"{self.dataset_key}_*"):
                if f.is_dir():
                    self.tp_to_flow_prefix_map[f.name] = [
                        (f"Smurf_Flows_gap{p1}/{f.name}",
                         f"Smurf_Flows_gap{p2}/{f.name}")
                        for (p1, p2) in itertools.combinations(pairs, 2)
                    ]

    def make_key(self, fkey, fid, type='rgb'):
        return f'{fkey}_{type}_{fid:0>3d}{self.type_2_suffix[type]}'

    def get_file(self, fkey, fid, type):
        key = self.make_key(fkey, fid, type)
        prefix = Path(self.path) / fkey
        if self.cache_path is not None:
            cache_prefix = Path(self.cache_path) / self.dataset_key / self.split / fkey
            if (cache_prefix / key).exists():
                if type == 'fwd' or type == 'bwd':
                    return np.load(cache_prefix / key)['arr_0']
                else:
                    im = Image.open(cache_prefix / key)
                    im.load()
                    return np.array(im)
        if type == 'fwd' or type == 'bwd':
            r = np.load(prefix / key)['arr_0']
            if self.cache_path is not None:
                cache_prefix = Path(self.cache_path) / self.dataset_key / self.split / fkey
                cache_prefix.mkdir(exist_ok=True, parents=True)
                np.savez(cache_prefix / key, r)
        else:
            im = Image.open(prefix / key)
            im.load()
            r = np.array(im)
            if self.cache_path is not None:
                cache_prefix = Path(self.cache_path) / self.dataset_key / self.split / fkey
                cache_prefix.mkdir(exist_ok=True, parents=True)
                im.save(cache_prefix / key)
        return r



    def __len__(self):
        if self.num_frames is None or self.num_frames > 1:
            return len(self.seq_index)
        return len(self.index)

    def __getitem__(self, idx):
        if self.random:
            idx = random.randrange(len(self))
        if self.num_frames == 1:
            prefix, fkey, fid = self.index[idx]
            return self.get_sample(prefix, fkey, fid)

        seq, sf = self.seq_index[idx]
        frames = []
        nf = self.num_frames or (max(self.seq_fid_2_idx[seq]) + 1)
        for fi in range(nf):
            prefix, fkey, fid = self.index[self.seq_fid_2_idx[seq][sf+fi]]
            frame = self.get_sample(prefix, fkey, fid)
            frames.append(frame[0])

        dataset_dict = {}
        for k in frames[0].keys():
            if torch.is_tensor(frames[0][k]):
                dataset_dict[k] = torch.stack([f[k] for f in frames])
            else:
                dataset_dict[k] = frames[0][k]
        return [dataset_dict]

    def get_seq_fid(self, seq, fid):
        idx = self.inv_index[(seq, fid)]
        prefix, fkey, fid = self.index[idx]
        return self.get_sample(prefix, fkey, fid)

    def get_sample(self, prefix, fkey, fid):

        gap = 'gap1'
        dataset_dicts = []
        dataset_dict = {}

        # start_frame = 3
        # tp = Path(flosplit[-2] + '.tar')
        # index = self.dataset.tpstr_to_index_map[str(tp)]
        # fid = start_frame+int(flosplit[-1].split('.')[0])
        rgb = torch.from_numpy(self.get_file(fkey, fid, 'rgb')).to(torch.float32).permute(2, 0, 1)
        h, w, = rgb.shape[-2:]
        if (h, w) != self.resolution:
            rgb = Fvt.resize(rgb, self.resolution, interpolation=Fvt.InterpolationMode.BICUBIC).clamp(min=0., max=255.)

        if self.gt_flow:
            flow = torch.from_numpy(self.get_file(fkey, fid, 'fwd')).to(torch.float32).permute(2, 0, 1)
            flow = flow[[1,0]]  # MOVi flows are VU instead of UV. This reverses it.
            if (h, w) != self.resolution:
                H, W = flow.shape[-2:]
                flow = Fvt.resize(flow, self.resolution, interpolation=Fvt.InterpolationMode.NEAREST)
                flow[:, :, 0] = flow[:, :, 0] * self.resolution[1] / W
                flow[:, :, 1] = flow[:, :, 1] * self.resolution[0] / H

            flo0 = flow
            if self.two_flow:
                flow = torch.from_numpy(self.get_file(fkey, fid, 'bwd')).to(torch.float32).permute(2, 0, 1)
                flow = flow[[1,0]]
                if (h, w) != self.resolution:
                    H, W = flow.shape[-2:]
                    flow = Fvt.resize(flow, self.resolution, interpolation=Fvt.InterpolationMode.NEAREST)
                    flow[:, :, 0] = flow[:, :, 0] * self.resolution[1] / W
                    flow[:, :, 1] = flow[:, :, 1] * self.resolution[0] / H
                flo1 = flow
        else:
            flos = self.tp_to_flow_prefix_map[fkey]
            if self.random:
                flos_prefix_0, flos_prefix_1 = random.choice(flos)
            else:
                flos_prefix_0, flos_prefix_1 = flos[0]
            gap = flos_prefix_0.split('/')[0].replace('Smurf_Flows_', '')
            flop0 = os.path.join(self.flow_dir, flos_prefix_0, f'{fkey}_rgb_{fid:0>3d}.flo')
            flo0 = torch.from_numpy(read_flow(str(flop0), self.resolution, self.to_rgb))
            if self.two_flow:
                flop1 = os.path.join(self.flow_dir, flos_prefix_1, f'{fkey}_rgb_{fid:0>3d}.flo')
                flo1 = torch.from_numpy(read_flow(str(flop1), self.resolution, self.to_rgb))

        dataset_dict["category"] = fkey
        dataset_dict['frame_id'] = fid
        dataset_dict['gap'] = gap

        sem_seg_gt = torch.from_numpy(self.get_file(fkey, fid, 'ano')).long().view(128, 128)
        if (h, w) != self.resolution:
            sem_seg_gt = Fvt.resize(sem_seg_gt[None], self.resolution, interpolation=Fvt.InterpolationMode.NEAREST)[0]

        H, W = flo0.shape[-2:]
        image_shape = (H, W)  # h, w

        if self.norm_flow:
            flo0 = flo0 / ((flo0 ** 2).sum(0).max().sqrt() + 1e-6)
            if self.two_flow:
                flo1 = flo1 / ((flo1 ** 2).sum(0).max().sqrt() + 1e-6)

        flo0 = flo0.clip(-self.flow_clip, self.flow_clip)
        if self.two_flow:
            flo1 = flo1.clip(-self.flow_clip, self.flow_clip)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo0.view(2, H, W)
        dataset_dict['height'] = image_shape[0]
        dataset_dict['width'] = image_shape[1]
        if self.two_flow:
            dataset_dict["flow_2"] = flo1.view(2, H, W)
        dataset_dict["rgb"] = rgb.view(3, H, W)

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
            dataset_dicts.append(dataset_dict)

        return dataset_dicts
