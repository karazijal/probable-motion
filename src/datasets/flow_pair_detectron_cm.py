import itertools
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fvt
import torchvision

from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from torch.utils.data import Dataset

from datasets.clevrmov import FlatCLEVRMOV
from utils.data import read_flow

logger = logging.getLogger('unsup_vidseg')


class FlowPairCMDetectron(Dataset):
    def __init__(self, split, pairs, flow_dir, res, resolution, to_rgb=False, size_divisibility=None, first=1000,
                 prefix=None, cache_path=None, two_flow=False, gt_flow=False, with_clevr=False, single_sample=False,
                 flow_clip=float('inf'), norm=False, ccrop=True, filter=False, num_frames=1, no_lims=False, first_frame_only=False,darken=False):
        self.to_rgb = to_rgb
        self.num_frames = num_frames
        self.resolution = resolution
        self.size_divisibility = size_divisibility
        self.ignore_label = -1
        self.gt_flow = gt_flow
        self.two_flow = two_flow
        self.random = split == 'train'
        min_frame = 2
        max_frame = 57
        if no_lims:
            min_frame = 0
            max_frame = 9999999
        if first_frame_only:
            min_frame = 1
            max_frame = 4

        self.dataset = FlatCLEVRMOV(
            split=split,
            load_clevr=with_clevr,
            load_mask=True,
            load_meta=False,
            load_fwd_flow=True,
            load_bwd_flow=True,
            prefix=prefix,
            seq_len=1,
            random_offset=True,
            transforms=lambda x: x,
            first=first if not no_lims else 999999,
            min_frame=min_frame,
            max_frame=max_frame,
            single_sample_flow=single_sample,
            cache_path=cache_path,
            filter=filter,
            no_val=no_lims,
        )
        self.darken = darken
        # self.dataset.tpstr_to_index_map = {str(tp.name): ind for (tp, ind) in self.dataset.inds.values()}
        # self.dataset.bias = 0
        # self.dataset.limit = len(self.dataset.inds)
        self.flow_dir = flow_dir or str(Path(prefix) / 'raft_new')
        self.data_dir = [flow_dir, flow_dir, flow_dir]
        self.tp_to_flow_prefix_map = {}
        if not self.gt_flow:
            for (tp, _) in self.dataset.inds.values():
                self.tp_to_flow_prefix_map[tp] = [
                    (f"Flows_gap{p1}/{res}/{tp.name.replace('.tar', '')}",
                     f"Flows_gap{p2}/{res}/{tp.name.replace('.tar', '')}")
                    for (p1, p2) in itertools.combinations(pairs, 2)
                ]

        logger.info(f"CLEVRMOV {split}: Prefix={prefix}, first {first} videos")
        self.flow_clip = flow_clip
        self.norm_flow = norm
        self.ccrop = ccrop
        if self.ccrop:
            self.resolution = (128, 128)
        if self.num_frames is None or self.num_frames > 1:
            self.seq_fid_2_idx = defaultdict(lambda: defaultdict(int))
            for idx in range(len(self.dataset)):
                tp, index, fid = self.dataset.idx_2_tarpath_index_frame_id(idx)
                seq = tp.name.replace('.tar', '')
                self.seq_fid_2_idx[seq][fid] = idx
            self.index = []
            for seq in self.seq_fid_2_idx:
                min_fid = min(self.seq_fid_2_idx[seq])
                max_fid = max(self.seq_fid_2_idx[seq])
                if self.num_frames is None:
                    self.index.append((seq, min_fid))
                else:
                    for fi in range(min_fid, max_fid + 1 - self.num_frames, 1 if self.random else self.num_frames):
                        self.index.append((seq, fi))

    def __len__(self):
        if self.num_frames is None or self.num_frames > 1:
            return len(self.index)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.random:
            idx = random.randrange(len(self))
        if self.num_frames == 1:
            tp, index, fid = self.dataset.idx_2_tarpath_index_frame_id(idx)
            return self.get_sample(tp, index, fid)

        seq, sf = self.index[idx]
        frames = []
        nf = self.num_frames or (max(self.seq_fid_2_idx[seq]) + 1)
        for fi in range(nf):
            tp, index, fid = self.dataset.idx_2_tarpath_index_frame_id(self.seq_fid_2_idx[seq][sf+fi])
            frame = self.get_sample(tp, index, fid)
            frames.append(frame[0])

        dataset_dict = {}
        for k in frames[0].keys():
            if torch.is_tensor(frames[0][k]):
                dataset_dict[k] = torch.stack([f[k] for f in frames])
            else:
                dataset_dict[k] = [f[k] for f in frames]
        return [dataset_dict]


    def get_seq_fid(self, seq, fid):
        idx = self.dataset.seq_fid_to_idx_map[(seq, fid)]
        tp, index, fid = self.dataset.idx_2_tarpath_index_frame_id(idx)
        return self.get_sample(tp, index, fid)

    def get_sample(self, tp, index, fid):
        gap = 'gap1'
        dataset_dicts = []
        dataset_dict = {}

        # start_frame = 3
        # tp = Path(flosplit[-2] + '.tar')
        # index = self.dataset.tpstr_to_index_map[str(tp)]
        # fid = start_frame+int(flosplit[-1].split('.')[0])
        rgb = self.dataset.get_file(tp,
                                    index,
                                    self.dataset.File.RGB,
                                    frame_id=fid)[:3]
        if self.darken:
            rgb = Fvt.adjust_brightness(rgb[None] / 255., 0.6165)[0] * 255.

        if self.ccrop:
            rgb = Fvt.center_crop(rgb, (192, 192))
        c, h, w = rgb.shape
        if (h, w) != tuple(self.resolution):
            rgb = Fvt.resize(rgb, self.resolution, interpolation=Fvt.InterpolationMode.BICUBIC)
        rgb = torch.clamp((rgb.to(torch.float32) / 255.0 - .5) * 2, min=-1., max=1)

        if self.gt_flow:
            flow = self.dataset.get_file(tp,
                                         index,
                                         self.dataset.File.FWD_FLOW,
                                         frame_id=fid)[:3]
            flow[0] = -flow[0]  # Flip x direction
            if self.ccrop:
                flow = Fvt.center_crop(flow, (192, 192))
            flow = einops.rearrange(flow, 'c h w -> h w c')
            h, w, _ = flow.shape
            if (h, w) != tuple(self.resolution):
                flow = cv2.resize(flow.numpy(), (self.resolution[1], self.resolution[0]),
                                  interpolation=cv2.INTER_NEAREST)
                flow[:, :, 0] = flow[:, :, 0] * self.resolution[1] / w
                flow[:, :, 1] = flow[:, :, 1] * self.resolution[0] / h
            else:
                flow = flow.numpy()
            flo0 = flow
            if self.two_flow:
                flow = self.dataset.get_file(tp,
                                             index,
                                             self.dataset.File.BWD_FLOW,
                                             frame_id=fid)[:3]
                flow[0] = -flow[0]  # Flip x direction
                if self.ccrop:
                    flow = Fvt.center_crop(flow, (192, 192))
                flow = einops.rearrange(flow, 'c h w -> h w c')
                h, w, _ = flow.shape
                if (h, w) != tuple(self.resolution):
                    flow = cv2.resize(flow.numpy(), (self.resolution[1], self.resolution[0]),
                                      interpolation=cv2.INTER_NEAREST)
                    flow[:, :, 0] = flow[:, :, 0] * self.resolution[1] / w
                    flow[:, :, 1] = flow[:, :, 1] * self.resolution[0] / h
                else:
                    flow = flow.numpy()
                flo1 = flow
        else:
            flos = self.tp_to_flow_prefix_map[tp]
            if self.random:
                flos_prefix_0, flos_prefix_1 = random.choice(flos)
            else:
                flos_prefix_0, flos_prefix_1 = flos[0]
            gap = flos_prefix_0.split('/')[0].replace('Flows_', '')
            flop0 = os.path.join(self.flow_dir, flos_prefix_0, f'{fid - 3:0>4d}.flo')
            flo0 = einops.rearrange(read_flow(str(flop0), self.resolution, self.to_rgb), 'c h w -> h w c')
            if self.two_flow:
                flop1 = os.path.join(self.flow_dir, flos_prefix_1, f'{fid - 3:0>4d}.flo')
                flo1 = einops.rearrange(read_flow(str(flop1), self.resolution, self.to_rgb), 'c h w -> h w c')

        dataset_dict["category"] = tp.stem
        dataset_dict['frame_id'] = fid - 3
        dataset_dict['gap'] = gap

        d2_utils.check_image_size(dataset_dict, flo0)
        sem_seg = self.dataset.get_file(tp, index, self.dataset.File.MASK, frame_id=fid)[0]
        if self.ccrop:
            sem_seg = Fvt.center_crop(sem_seg, (192, 192))
        sem_seg_gt = cv2.resize(sem_seg.numpy(),
                                (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
        if sem_seg_gt.ndim == 3:
            sem_seg_gt = sem_seg_gt[:, :, 0]
        if sem_seg_gt.max() == 255:
            sem_seg_gt[sem_seg_gt == 255] = 1

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
        if self.to_rgb:
            flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1))) / 2 + .5
            flo0 = flo0 * 255
            if self.two_flow:
                flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1))) / 2 + .5
                flo1 = flo1 * 255
        else:
            flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1)))
            if self.two_flow:
                flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1)))

            if self.norm_flow:
                flo0 = flo0 / ((flo0 ** 2).sum(0).max().sqrt() + 1e-6)
                if self.two_flow:
                    flo1 = flo1 / ((flo1 ** 2).sum(0).max().sqrt() + 1e-6)
            flo0 = flo0.clip(-self.flow_clip, self.flow_clip)
            if self.two_flow:
                flo1 = flo1.clip(-self.flow_clip, self.flow_clip)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb)).float() / 2 + .5
        rgb = rgb * 255

        rgb = rgb.float()
        flo0 = flo0.float()
        if self.two_flow:
            flo1 = flo1.float()

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (flo0.shape[-2], flo0.shape[-1])
            padding_size = [
                0,
                int(self.size_divisibility * math.ceil(image_size[1] // self.size_divisibility)) - image_size[1],
                0,
                int(self.size_divisibility * math.ceil(image_size[0] // self.size_divisibility)) - image_size[0],
            ]
            flo0 = F.pad(flo0, padding_size, value=0).contiguous()
            if self.two_flow:
                flo1 = F.pad(flo1, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (flo0.shape[-2], flo0.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo0
        if self.two_flow:
            dataset_dict["flow_2"] = flo1
        dataset_dict["rgb"] = rgb
        dataset_dict["flow_rgb"] = torchvision.utils.flow_to_image(flo0).float()

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

            fg_bg_classes = torch.tensor(classes, dtype=torch.int64)
            fg_bg_classes = torch.ones_like(fg_bg_classes)
            fg_bg_classes[0] = 0
            instances.gt_classes = fg_bg_classes

            dataset_dict["instances"] = instances
            dataset_dicts.append(dataset_dict)

        return dataset_dicts
