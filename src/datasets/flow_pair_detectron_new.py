import itertools
import math
import shutil
import filelock
from functools import lru_cache
from pathlib import Path

import detectron2.data.transforms as DT
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from torch.utils.data import Dataset

import utils
from utils.data import read_flow2, UnindetifedFlowError

logger = utils.log.getLogger(__name__)


@lru_cache(maxsize=None)
def get_frames(fwd_path_str, bwd_path_str, rgb_path_str, seq, rgb_suffix='.jpg'):
    fwd = Path(fwd_path_str)
    bwd = Path(bwd_path_str)
    rgb_path = Path(rgb_path_str)

    fwd_flows = {}
    for f in (fwd / seq).glob(f'*.flo'):
        fid = f.stem.split('_')[-1].lstrip('0')
        fid = int(fid) if fid else 0
        fwd_flows[fid] = f
    bwd_flows = {}
    for f in (bwd / seq).glob(f'*.flo'):
        fid = f.stem.split('_')[-1].lstrip('0')
        fid = int(fid) if fid else 0
        bwd_flows[fid] = f
    rgbs = {}
    for f in (rgb_path / seq).glob(f'*{rgb_suffix}'):
        fid = f.stem.split('_')[-1].lstrip('0')
        fid = int(fid) if fid else 0
        rgbs[fid] = f
    # intersect all the flows and rgbs
    fids = sorted(list(set(fwd_flows.keys()) & set(bwd_flows.keys()) & set(rgbs.keys())))
    return {fid: (rgbs[fid], fwd_flows[fid], bwd_flows[fid]) for fid in fids}

def get_lock(path):
    path = Path(path)
    return filelock.FileLock(str(path.with_suffix('.lock')))

class FlowPairNewDetectron(Dataset):
    def __init__(self,
                 gaps,
                 resolution,
                 prefix=None,
                 cache_path=None,
                 two_flow=False,
                 flow_clip=float('inf'),
                 norm=False,
                 num_frames=1,
                 res='480p',
                 sequences=None,
                 size_divisibility=-1,
                 return_original=False,
                 rgb_dir='JPEGImages',
                 rgb_suffix='.jpg',
                 msk_dir='Annotations',
                 flows_prefix='Flows_gap',
                 pseudo_index=None
                 ):
        self.return_original = return_original
        self.load_mg = False
        self.two_flow = two_flow
        self.size_divisibility = size_divisibility

        self.ignore_label = -1
        self.flow_clip = flow_clip
        self.norm_flow = norm
        self.resolution = resolution
        self.transforms = DT.AugmentationList([
            DT.Resize(self.resolution, interp=Image.BICUBIC),
        ])
        self.num_frames = num_frames if num_frames is not None and num_frames > 1 else 1

        self.prefix = Path(prefix)
        self.data_dir = [str(prefix)]
        self.cache_path = cache_path if cache_path is None else Path(cache_path)

        if isinstance(res, (tuple, list)):
            self.res_rgb, self.res_mask, self.res_flow = res
        else:
            self.res_rgb, self.res_mask, self.res_flow = res, res, res

        self.rgb_path = Path(prefix) / rgb_dir / self.res_rgb
        self.mask_path = Path(prefix) / msk_dir / self.res_mask

        fwd_gaps = [x for x in gaps if x > 0]
        bwd_gaps = [x for x in gaps if x < 0]

        flo_gaps = list(itertools.product(fwd_gaps, bwd_gaps))
        if self.num_frames > 1:
            flo_gaps = [(g, -g) for g in fwd_gaps]
            logger.info(f"Using {self.num_frames} frames. The frame gaps will be paired as {flo_gaps}")

        flow_path_pairs = [(a, b, Path(prefix) / f'{flows_prefix}{a}' / self.res_flow,
                            Path(prefix) / f'{flows_prefix}{b}' / self.res_flow) for a, b in flo_gaps]

        # get rgb sub-directories as sequence
        if pseudo_index is not None:
            sequences = list(pseudo_index.keys())
        if sequences is None:
            sequences = [x.name for x in self.rgb_path.glob('*') if x.resolve().is_dir()]
        self.sequences = sequences

        # check these folders exist for masks and flows
        for seq in sequences:
            if not (self.rgb_path / seq).exists():
                raise ValueError(f'{self.rgb_path / seq} does not exist')
            if not (self.mask_path / seq).exists():
                logger.warning(f"Missing mask folder {seq}")
            for path_pair in flow_path_pairs:
                if not (path_pair[2] / seq).resolve().exists():
                    raise ValueError(f'Missing {path_pair[2] / seq}')
                if not (path_pair[3] / seq).resolve().exists():
                    raise ValueError(f'Missing {path_pair[3] / seq}')

        self.index_map = {}
        for a, b, fwd, bwd in flow_path_pairs:
            self.index_map[(a, b)] = {}
            for seq in sequences:
                frame_map = get_frames(str(fwd), str(bwd), str(self.rgb_path), seq, rgb_suffix=rgb_suffix)
                if len(frame_map) == 0:
                    logger.warning(f'No frames found for {seq}')
                    continue
                self.index_map[(a, b)][seq] = frame_map

        self.index = []
        for (fwd_gap, bwd_gap) in self.index_map:
            for seq in self.index_map[(fwd_gap, bwd_gap)]:
                if self.num_frames == 1:
                    if pseudo_index is None:
                        self.index.extend(
                            (fwd_gap, bwd_gap, seq, fid) for fid in self.index_map[(fwd_gap, bwd_gap)][seq].keys())
                    else:
                        self.index.extend(
                            (fwd_gap, bwd_gap, seq, fid) for fid in sorted(pseudo_index[seq]))
                else:
                    if pseudo_index is None:
                        min_fid = min(self.index_map[(fwd_gap, bwd_gap)][seq].keys())
                        max_fid = max(self.index_map[(fwd_gap, bwd_gap)][seq].keys())
                    else:
                        min_fid = min(pseudo_index[seq])
                        max_fid = max(pseudo_index[seq])
                    for start_fid in range(min_fid, max_fid + 1 - self.num_frames, fwd_gap):
                        self.index.append((fwd_gap, bwd_gap, seq, start_fid))
                        

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Prevent stacking of frames which will inject time axis
        if self.num_frames == 1:
            return self.get_item(*self.index[idx])

        frames = []
        fwd_gap, bwd_gap, seq, start_fid = self.index[idx]
        for fid in range(self.num_frames):
            f = self.get_item(fwd_gap, bwd_gap, seq, start_fid + fid)
            frames.append(f[0])

        dataset_dict = {}
        for k in frames[0].keys():
            if torch.is_tensor(frames[0][k]):
                dataset_dict[k] = torch.stack([f[k] for f in frames])
            else:
                dataset_dict[k] = frames[0][k]
        return [dataset_dict]

    def __lock(self, path):
        return get_lock(path)

    def __maybe_cache_path(self, file_path):
        if self.cache_path is None:
            return file_path

        max_attempts = 3
        cache_path = self.cache_path / file_path.relative_to(self.prefix)

        # quick check
        if cache_path.exists() and cache_path.is_file():
            return cache_path

        cache_part = cache_path.parent / (cache_path.name + '.part')
        cache_part.parent.mkdir(parents=True, exist_ok=True)
        try:
            with get_lock(cache_part).acquire(blocking=True):
                size = file_path.stat().st_size
                shutil.copy(file_path, cache_part)
                attempts = 1
                while attempts < max_attempts and cache_part.stat().st_size != size:
                    attempts += 1
                    shutil.copy(file_path, cache_part)
                if attempts == max_attempts:
                    logger.error(f'Failed to copy {file_path} to {cache_path}; returning original path')
                    if cache_part.exists():
                        cache_part.unlink()
                    return file_path
                if attempts > 1:
                    logger.warning(f'Copying {file_path} to {cache_path} required {attempts} attempts')
                cache_part.rename(cache_path)  # Write a file to indicate it's done
        except (filelock.Timeout, OSError, IOError):
            logger.error(f'Failed make local copy for {file_path}; using original')
            return file_path
        return cache_path

    def __maybe_cache_evict(self, cache_path):
        if self.cache_path is None:
            return True
        cache_path.relative_to(self.cache_path)  # raises ValueError if not in cache_path
        cache_part = cache_path.parent / (cache_path.name + '.part')
        lock = get_lock(cache_part)
        try:
            with lock.acquire(blocking=True):
                cache_path.unlink()
                return True
        except (filelock.Timeout, OSError, IOError):
            logger.error(f'Failed to delete {cache_path}')
            return False


    def get_image(self, img_path):
        maybe_cached_path = self.__maybe_cache_path(img_path)
        try:
            return d2_utils.read_image(str(maybe_cached_path))
        except UnidentifiedImageError:
            logger.warning(f'Failed to read {maybe_cached_path}')
            if self.cache_path is not None and maybe_cached_path != img_path:
                self.__maybe_cache_evict(maybe_cached_path)
            return d2_utils.read_image(str(img_path))  # Try again from (maybe) remote

    def get_flow(self, flow_path):
        maybe_cached_path = self.__maybe_cache_path(flow_path)
        try:
            return read_flow2(str(maybe_cached_path), self.resolution, False, False)
        except UnindetifedFlowError:
            logger.warning(f'Failed to read {maybe_cached_path}')
            if self.cache_path is not None and maybe_cached_path != flow_path:
                self.__maybe_cache_evict(maybe_cached_path)
            return read_flow2(str(flow_path), self.resolution, False, False) # Try again from (maybe) remote

    def get_item(self, fwd_gap, bwd_gap, seq, fid):
        dataset_dicts = []
        dataset_dict = {
            'category': seq,
            'frame_id': fid
        }

        rgb_path = self.index_map[(fwd_gap, bwd_gap)][seq][fid][0]
        rgb = self.get_image(rgb_path).astype(np.float32)
        if len(rgb.shape) == 3:
            rgb = rgb[..., :3]
        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0., 255.))).float()

        fwd_path = self.index_map[(fwd_gap, bwd_gap)][seq][fid][1]
        flo0 = einops.rearrange(self.get_flow(fwd_path), 'c h w -> h w c')

        bwd_path = self.index_map[(fwd_gap, bwd_gap)][seq][fid][2]
        flo1 = einops.rearrange(self.get_flow(bwd_path), 'c h w -> h w c')


        # if self.ccrop:
        #     h, w, _ = np.shape(rgb)
        #     s = int(min(h, w) * self.crop_frac)
        #     rgb = rgb[(h - s) // 2: (h - s) // 2 + s, (w - s) // 2: (w - s) // 2 + s]

        # print('not here', rgb.min(), rgb.max())
        input = DT.AugInput(rgb)

        # Apply the augmentation:
        preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        rgb = input.image
        # if self.photometric_aug:
        #     rgb_aug = Image.fromarray(rgb.astype(np.uint8))
        #     rgb_aug = self.photometric_aug(rgb_aug)
        #     rgb_aug = d2_utils.convert_PIL_to_numpy(rgb_aug, 'RGB')
        #     rgb_aug = np.transpose(rgb_aug, (2, 0, 1)).astype(np.float32)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0., 255.)

        d2_utils.check_image_size(dataset_dict, flo0)


        gt_path = (self.mask_path / seq / rgb_path.stem).with_suffix('.png')
        if self.res_rgb != self.res_mask:
            gt_path = Path(str(gt_path).replace(str(self.res_rgb), str(self.res_mask)))

        if gt_path.exists():
            sem_seg_gt_ori = self.get_image(gt_path)
            # if self.ccrop:
            #     h, w, *_ = np.shape(sem_seg_gt_ori)
            #     s = min(h, w)
            #     sem_seg_gt_ori = sem_seg_gt_ori[(h - s) // 2: (h - s) // 2 + s, (w - s) // 2: (w - s) // 2 + s]
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt_ori)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        mg_path = (self.prefix / 'motiongrouping' / self.res_rgb / seq / rgb_path.stem).with_suffix('.png')
        if self.load_mg and mg_path.exists():
            mg_seg_gt = self.get_image(mg_path)
            # if self.ccrop:
            #     h, w, *_ = np.shape(mg_seg_gt)
            #     s = int(min(h, w) * self.crop_frac)
            #     mg_seg_gt = mg_seg_gt[(h - s) // 2: (h - s) // 2 + s, (w - s) // 2: (w - s) // 2 + s]
            mg_seg_gt = preprocessing_transforms.apply_segmentation(mg_seg_gt)
            if mg_seg_gt.ndim == 3:
                mg_seg_gt = mg_seg_gt[:, :, 0]
            if mg_seg_gt.max() == 255:
                mg_seg_gt[mg_seg_gt == 255] = 1
        else:
            mg_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
        flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1)))
        flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1)))

        if self.norm_flow:
            flo0 = flo0 / (flo0 ** 2).sum(0).max().sqrt()
            flo1 = flo1 / (flo1 ** 2).sum(0).max().sqrt()
        flo0 = flo0.clip(-self.flow_clip, self.flow_clip)
        flo1 = flo1.clip(-self.flow_clip, self.flow_clip)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))
        if mg_seg_gt is not None:
            mg_seg_gt = torch.as_tensor(mg_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (flo0.shape[-2], flo0.shape[-1])
            padding_size = [
                0,
                int(self.size_divisibility * math.ceil(image_size[1] // self.size_divisibility)) - image_size[1],
                0,
                int(self.size_divisibility * math.ceil(image_size[0] // self.size_divisibility)) - image_size[0],
            ]
            flo0 = F.pad(flo0, padding_size, value=0).contiguous()
            flo1 = F.pad(flo1, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            # if self.photometric_aug:
            #     rgb_aug = F.pad(rgb_aug, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if mg_seg_gt is not None:
                mg_seg_gt = F.pad(mg_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (flo0.shape[-2], flo0.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo0
        dataset_dict["flow_2"] = flo1
        dataset_dict["rgb"] = rgb
        if self.return_original:
            dataset_dict["original_rgb"] =  F.interpolate(
                original_rgb[None],
                mode='bicubic',
                size=sem_seg_gt_ori.shape[-2:],
                align_corners=False).clip(0., 255.)[0]

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            if self.return_original:
                dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()

        if mg_seg_gt is not None:
            dataset_dict["mg_seg"] = mg_seg_gt.long()

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
