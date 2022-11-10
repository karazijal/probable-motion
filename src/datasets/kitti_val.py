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


def get_lock(path):
    path = Path(path)
    return filelock.FileLock(str(path.with_suffix(".lock")))


class KITTI_VAL(Dataset):
    def __init__(
        self,
        resolution,
        prefix=None,
        cache_path=None,
        size_divisibility=-1,
        return_original=False,
    ):
        self.return_original = return_original
        self.size_divisibility = size_divisibility

        self.ignore_label = -1
        self.resolution = resolution
        self.transforms = DT.AugmentationList(
            [
                DT.Resize(self.resolution, interp=Image.BICUBIC),
            ]
        )

        self.prefix = Path(prefix)
        self.data_dir = [str(prefix)]
        self.cache_path = cache_path if cache_path is None else Path(cache_path)

        self.index = []
        for ano_path in (self.prefix / "instance").glob("*.png"):
            msk_path = ano_path.resolve()
            rgb_path = (ano_path.parent.parent / "image_2" / ano_path.name).resolve()
            if (
                msk_path.exists()
                and rgb_path.exists()
                and msk_path.is_file()
                and rgb_path.is_file()
            ):
                self.index.append((rgb_path, msk_path))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Prevent stacking of frames which will inject time axis
        return self.get_item(idx)

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

        cache_part = cache_path.parent / (cache_path.name + ".part")
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
                    logger.error(
                        f"Failed to copy {file_path} to {cache_path}; returning original path"
                    )
                    if cache_part.exists():
                        cache_part.unlink()
                    return file_path
                if attempts > 1:
                    logger.warning(
                        f"Copying {file_path} to {cache_path} required {attempts} attempts"
                    )
                cache_part.rename(cache_path)  # Write a file to indicate it's done
        except (filelock.Timeout, OSError, IOError):
            logger.error(f"Failed make local copy for {file_path}; using original")
            return file_path
        return cache_path

    def __maybe_cache_evict(self, cache_path):
        if self.cache_path is None:
            return True
        cache_path.relative_to(
            self.cache_path
        )  # raises ValueError if not in cache_path
        cache_part = cache_path.parent / (cache_path.name + ".part")
        lock = get_lock(cache_part)
        try:
            with lock.acquire(blocking=True):
                cache_path.unlink()
                return True
        except (filelock.Timeout, OSError, IOError):
            logger.error(f"Failed to delete {cache_path}")
            return False

    def get_image(self, img_path):
        maybe_cached_path = self.__maybe_cache_path(img_path)
        try:
            return d2_utils.read_image(str(maybe_cached_path))
        except UnidentifiedImageError:
            logger.warning(f"Failed to read {maybe_cached_path}")
            if self.cache_path is not None and maybe_cached_path != img_path:
                self.__maybe_cache_evict(maybe_cached_path)
            return d2_utils.read_image(str(img_path))  # Try again from (maybe) remote

    def decode_sem_seg(self, sem_seg, force_resolution=(374, 1241)):
        instance_gt = sem_seg % 256
        semantic_gt = sem_seg // 256
        segs = np.unique(instance_gt)
        mask = np.zeros(instance_gt.shape, dtype=np.uint8)
        for i, seg in enumerate(segs):
            if seg == self.ignore_label:
                continue
            mask[instance_gt == seg] = i
        mask = F.interpolate(
            torch.from_numpy(mask)[None, None], size=force_resolution, mode="nearest"
        )[0, 0].numpy()
        # print('MASK', mask.shape)
        return mask

    def get_item(self, idx):
        dataset_dicts = []
        rgb_path, msk_path = self.index[idx]
        dataset_dict = {"category": msk_path.stem, "frame_id": idx}
        rgb = self.get_image(rgb_path).astype(np.float32)
        if len(rgb.shape) == 3:
            rgb = rgb[..., :3]
        original_rgb = torch.as_tensor(
            np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0.0, 255.0))
        ).float()
        original_rgb = F.interpolate(
            original_rgb[None], size=(374, 1241), mode="bilinear", align_corners=False
        )[0]
        rgb = original_rgb.permute(1, 2, 0).numpy()
        # print("RGB", rgb.shape)

        # if self.ccrop:
        #     h, w, _ = np.shape(rgb)
        #     s = int(min(h, w) * self.crop_frac)
        #     rgb = rgb[(h - s) // 2: (h - s) // 2 + s, (w - s) // 2: (w - s) // 2 + s]

        # print('not here', rgb.min(), rgb.max())
        input = DT.AugInput(rgb)

        # Apply the augmentation:
        preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        rgb = input.image
        # print("TF", preprocessing_transforms[0].h, preprocessing_transforms[0].w)
        # if self.photometric_aug:
        #     rgb_aug = Image.fromarray(rgb.astype(np.uint8))
        #     rgb_aug = self.photometric_aug(rgb_aug)
        #     rgb_aug = d2_utils.convert_PIL_to_numpy(rgb_aug, 'RGB')
        #     rgb_aug = np.transpose(rgb_aug, (2, 0, 1)).astype(np.float32)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0.0, 255.0)

        measurement_res = (96, 320)

        gt_path = msk_path
        if gt_path.exists():
            sem_seg_gt_ori = self.get_image(gt_path)
            sem_seg_gt_ori = self.decode_sem_seg(sem_seg_gt_ori)
            # if self.ccrop:
            #     h, w, *_ = np.shape(sem_seg_gt_ori)
            #     s = min(h, w)
            #     sem_seg_gt_ori = sem_seg_gt_ori[(h - s) // 2: (h - s) // 2 + s, (w - s) // 2: (w - s) // 2 + s]
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt_ori)

            sem_seg_gt = F.interpolate(
                torch.from_numpy(sem_seg_gt)[None, None],
                size=measurement_res,
                mode="nearest",
            )[0, 0].numpy()
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        rgb = torch.as_tensor(np.ascontiguousarray(rgb))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))

        flo0 = flo1 = torch.zeros((2, rgb.shape[-2], rgb.shape[-1]))

        if self.size_divisibility > 0:
            image_size = (rgb.shape[-2], rgb.shape[-1])
            padding_size = [
                0,
                int(
                    self.size_divisibility
                    * math.ceil(image_size[1] // self.size_divisibility)
                )
                - image_size[1],
                0,
                int(
                    self.size_divisibility
                    * math.ceil(image_size[0] // self.size_divisibility)
                )
                - image_size[0],
            ]
            flo0 = F.pad(flo0, padding_size, value=0).contiguous()
            flo1 = F.pad(flo1, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            # if self.photometric_aug:
            #     rgb_aug = F.pad(rgb_aug, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()

        image_shape = (flo0.shape[-2], flo0.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo0
        dataset_dict["flow_2"] = flo1
        dataset_dict["rgb"] = rgb
        if self.return_original:
            dataset_dict["original_rgb"] = F.interpolate(
                original_rgb[None],
                mode="bicubic",
                size=sem_seg_gt_ori.shape[-2:],
                align_corners=False,
            ).clip(0.0, 255.0)[0]

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            if self.return_original:
                dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()
        dataset_dict["height"] = measurement_res[0]
        dataset_dict["width"] = measurement_res[1]

        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'."
            )

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
                instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
                )
            else:
                masks = BitMasks(
                    torch.stack(
                        [
                            torch.from_numpy(np.ascontiguousarray(x.copy()))
                            for x in masks
                        ]
                    )
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
            dataset_dicts.append(dataset_dict)

        return dataset_dicts
