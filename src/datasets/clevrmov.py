import functools
import io
import itertools
import json
import logging
import tarfile
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

try:
    import OpenEXR

    OPENEXRAVAILABLE = True
except ImportError:
    print(f'Could not find openEXR; loading flow will error')
    OPENEXRAVAILABLE = False

LOGGER = logging.getLogger(__name__)


def open_exr(fpath):
    fpath = Path(fpath)
    with Path(fpath).open('rb') as inf:
        return exr_to_numpy(inf.read())


def exr_to_numpy(buffer):
    with io.BytesIO(buffer) as f:
        exr = OpenEXR.InputFile(f)
        h = exr.header()
        channels = h['channels']
        dw = h['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        npas = []
        if any(c in channels for c in 'RGBA'):
            # Decode in RGBA order skipping unknown -- ensures cannocical RGBA C1 C2 ... Order
            for c in 'RGBA':
                if c not in channels:
                    continue
                npas.append(np.frombuffer(exr.channel(c, channels[c].type), dtype=np.float32).reshape(size[1], size[0]))
            # Now decode remaingin channels
            for c in channels:
                if c in 'RGBA':
                    continue
                npas.append(np.frombuffer(exr.channel(c, channels[c].type), dtype=np.float32).reshape(size[1], size[0]))
        else:
            raise ValueError(f'No RGBA channels found in EXR channels: {channels}')
        return np.stack(npas, axis=-1)


def tar_indexed_get(index, key, prefix="", cache_dir=None):
    if key not in index:
        raise KeyError(f"Key {key} not in the index")
    start, length = index[key]
    p = Path(prefix) / Path(key).parent
    k = Path(key)

    cache_path = cache_dir

    if cache_path is not None:
        cached_path = Path(cache_path) / Path(key)
        if cached_path.exists() and cached_path.is_file():
            if k.suffix == ".png" or k.suffix == ".jpg":
                try:
                    return Image.open(cached_path)
                except UnidentifiedImageError:
                    cached_path.unlink(missing_ok=True)
                    return tar_indexed_get(index, key, prefix, cache_dir)
            elif k.suffix == ".json":
                return json.load(cached_path)
            elif k.suffix == '.exr' and OPENEXRAVAILABLE:
                with cached_path.open('rb') as inf:
                    buffer = inf.read()
                arr = exr_to_numpy(buffer)
                if '_flow_' in k.stem:
                    arr = arr[..., :2]  # Only first 2 channels for flow
                if '_depth_' in k.stem or '_mist_' in k.stem:
                    arr = arr[..., :1]  # Only single channel for depth
                return arr
            elif k.suffix == ".blend":
                with cached_path.open('rb') as inf:
                    buffer = inf.read()
                    return io.BytesIO(buffer)

    with Path(str(p) + ".tar").open("rb") as inf:
        inf.seek(start)
        buffer = inf.read(length)
        if cache_path is not None:
            cached_path = Path(cache_path) / Path(key)
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            with cached_path.open('wb') as outf:
                outf.write(buffer)

        if k.suffix == ".png" or k.suffix == ".jpg":
            return Image.open(io.BytesIO(buffer))
        elif k.suffix == ".json":
            return json.loads(buffer)
        elif k.suffix == '.exr' and OPENEXRAVAILABLE:
            arr = exr_to_numpy(buffer)
            if '_flow_' in k.stem:
                arr = arr[..., :2]  # Only first 2 channels for flow
            if '_depth_' in k.stem or '_mist_' in k.stem:
                arr = arr[..., :1]  # Only single channel for depth
            return arr
        elif k.suffix == ".blend":
            return io.BytesIO(buffer)
        else:
            raise ValueError(f"Unknown file type {k.suffix}")


def build_index(tar_path, index_path):
    tp = Path(tar_path)
    ip = Path(index_path)
    with tarfile.open(tp, mode="r:") as tarf:
        index = {"_t": tp.stat().st_mtime}
        for tari in tarf.getmembers():
            if not tari.isfile():
                continue
            index[tari.name] = (tari.offset_data, tari.size)
    try:
        with ip.open("w") as outf:
            json.dump(index, outf)
    except (IOError, OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(f"Could not save index due to {str(exc)}")
    return index


def get_index(tar_path, rebuild=False):
    tar_path = Path(tar_path)
    if tar_path.parent.name != 'tar':
        index_path = Path(str(tar_path) + f".index.json")
    else:
        tar_name = tar_path.name
        index_path = tar_path.parent.parent / 'json' / (str(tar_name) + f".index.json")
    try:
        if index_path.exists() or not rebuild:
            with index_path.open("r") as inf:
                index = json.load(inf)
        else:
            LOGGER.warning(f"Missing index for {tar_path.name}; Building")
            index = build_index(tar_path, index_path)
    except (IOError, OSError, json.JSONDecodeError) as exc:
        LOGGER.info(
            f"Could not read index for {tar_path.name}; Rebuilding; Reason {str(exc)}"
        )
        index = build_index(tar_path, index_path)
    return index


def index_with_bias_and_limit(idx, bias, limit):
    if idx >= 0:
        idx += bias
        if idx >= limit:
            raise IndexError()
    else:
        idx = limit + idx
        if idx < bias:
            raise IndexError()
    return idx


class ComposedTransformed:
    def __init__(self, *t):
        if len(t) == 1 and isinstance(t[0], (list, tuple)):
            t = t[0]
        self.t = t

    def __call__(self, *x):
        if len(x) == 1 and isinstance(x[0], (list, tuple)):
            x = x[0]
        for t in self.t:
            x = [tx(xx) for tx, xx in zip(t, x)]
        return x


def resize128(inp, mode="bilinear"):
    return torch.nn.functional.interpolate(
        inp,
        size=(128, 128),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )


DEFAULT_TRANSFORMS = ComposedTransformed(
    [
        tv.transforms.CenterCrop(192),
        tv.transforms.CenterCrop(192),
        tv.transforms.CenterCrop(192),
    ],
    [resize128, partial(resize128, mode="nearest"), partial(resize128, mode="nearest")],
)


@functools.cache
def find_data(prefix, first, verbose):
    inds = {}
    seq_to_tp_map = {}
    print(f"Indexing")
    for tp in tqdm(sorted(prefix.glob("*.tar")), desc=f"Indexing CLEVRMOV", disable=not verbose):
        ind = int(tp.stem[-6:])
        if first is not None and ind > first:
            continue
        inds[ind] = (tp, get_index(tp))
        seq_to_tp_map[tp.name.replace('.tar', '')] = tp
    return inds, seq_to_tp_map

class CLEVRMOV:
    """CLEVRMOV - MOVing CLEVR/ClevrTex dataset."""
    DEFAULT_PREFIX = "data/clevrmov"
    SPLITS = {"test": (0.0, 0.1), "val": (0.1, 0.2), "train": (0.2, 1.0), "__full_dataset__": (0.0, 1.0)}
    ## Val is test
    SPLITS_NO_VAL = {"test": (0.0, 0.1), "val": (0.0, 0.1), "train": (0.1, 1.0), "__full_dataset__": (0.0, 1.0)}

    __CLEVR_MODE = "clevr_"
    __SNG_FLOW_MODE = 'single_sample_'

    __FWD_FLOW = "fwd_flow_"
    __BWD_FLOW = "bwd_flow_"
    __MASK_V0 = "flat_"  # Deprecated
    __MASK_V1 = "sem_"

    class File(Enum):
        RGB = 0
        MASK = 1
        FWD_FLOW = 2
        BWD_FLOW = 3

    @property
    def mask_file_infix(self):
        r = []
        if self.load_clevr:
            r.append(self.__CLEVR_MODE)
        if self.mask_version == 'v0':
            r.append(self.__MASK_V0)
        else:
            r.append(self.__MASK_V1)
        return ''.join(r)

    @property
    def rgb_file_infix(self):
        if self.load_clevr:
            return self.__CLEVR_MODE
        return ""

    @property
    def fwd_flow_infix(self):
        r = []
        if self.load_clevr:
            r.append(self.__CLEVR_MODE)
        if self.single_sample_flow:
            r.append(self.__SNG_FLOW_MODE)
        r.append(self.__FWD_FLOW)
        return ''.join(r)

    @property
    def bwd_flow_infix(self):
        r = []
        if self.load_clevr:
            r.append(self.__CLEVR_MODE)
        if self.single_sample_flow:
            r.append(self.__SNG_FLOW_MODE)
        r.append(self.__BWD_FLOW)
        return ''.join(r)

    def __init__(
            self,
            split="train",
            load_clevr=False,
            load_mask=True,
            load_meta=False,
            load_fwd_flow=False,
            load_bwd_flow=False,
            prefix=None,
            cache_path=None,
            seq_len=1,
            random_offset=False,
            transforms=DEFAULT_TRANSFORMS,
            first=100,
            verbose=True,
            mask_version=None,
            single_sample_flow=False,
            no_val=False
    ):
        """
        CLEVRMOV dataset
        :param split:
        :param load_mask: Load segmentation masks
        :param load_meta: Load metadata :TODO:
        :param load_flow: Load flow (vector) images
        :param prefix: Dataset location prefix
        :param seq_len: Return sequences of lenth <seq_len>
        :param random_offset: Randomly sample within sequence
        :param transforms: Apply a data = transforms(data)
        :param first: Limit dataset up to <first> limit; set to None to disable
        """
        self.prefix = Path(prefix or self.DEFAULT_PREFIX)
        self.single_sample_flow = single_sample_flow
        self.mask_version = mask_version
        if self.prefix.name != 'tar':
            self.prefix = self.prefix / 'tar'
        
        if first is not None:
            print(f"Limiting dataset to {first} samples")

        self.inds, self.seq_to_tp_map = find_data(self.prefix, first, verbose)
        print(f"Found {len(self.inds)} values at {self.prefix}")

        self.cache_path = cache_path

        self.seq_len = seq_len
        self.random_offset = random_offset

        self.load_clevr = load_clevr
        self.load_meta = load_meta
        self.load_mask = load_mask
        self.load_fwd_flow = load_fwd_flow
        self.load_bwd_flow = load_bwd_flow

        self.split = split
        bias, limit = self.SPLITS[split]
        if no_val:
            bias, limit = self.SPLITS_NO_VAL[split]
        if isinstance(bias, float):
            bias = int(bias * len(self.inds))
        if isinstance(limit, float):
            limit = int(limit * len(self.inds))
        self.limit = limit
        self.bias = bias
        self.transforms = transforms

    def __len__(self):
        return self.limit - self.bias

    def _make_key(self, tp, file_enum, frame_id=3):
        suffix = '.png'
        if file_enum in {self.File.FWD_FLOW, self.File.BWD_FLOW}:
            suffix = '.exr'
        if file_enum == self.File.RGB:
            infix = self.rgb_file_infix
        elif file_enum == self.File.MASK:
            infix = self.mask_file_infix
        elif file_enum == self.File.FWD_FLOW:
            infix = self.fwd_flow_infix
        elif file_enum == self.File.BWD_FLOW:
            infix = self.bwd_flow_infix
        else:
            raise ValueError(f"Unsupported file type {file_enum}")
        tp = Path(tp)
        return tp.stem + "/" + tp.stem + f"_{infix}{frame_id:>04d}{suffix}"

    def get_file(self, tp, index, file_type, frame_id=3):
        key = self._make_key(tp, file_type, frame_id)
        img = torch.from_numpy(
            np.array(tar_indexed_get(index, key, prefix=self.prefix, cache_dir=self.cache_path))
        )
        if len(img.shape) == 3:
            img = img.permute(2, 0, 1)
        elif len(img.shape) == 2:
            img = img[None]
        return img

    def _num_frames(self, index):
        return max(int(k[-8:-4]) for k in index.keys() if k.endswith(".png"))

    def idx_2_tarpath_index_frame_id(self, idx):
        idx = index_with_bias_and_limit(idx, self.bias, self.limit)
        tp, index = self.inds[idx]
        num_frames = self._num_frames(index)
        if self.random_offset:
            start_frame = np.random.randint(3, num_frames - self.seq_len)
        else:
            start_frame = 3
        return tp, index, start_frame

    def __getitem__(self, idx):
        tp, index, start_frame = self.idx_2_tarpath_index_frame_id(idx)

        imgs = []
        masks = []
        fwd_flows = []
        bwd_flows = []
        meta_key = tp.stem + "/" + tp.stem + ".json"
        if self.load_meta:
            meta = tar_indexed_get(index, meta_key, prefix=self.prefix, cache_dir=self.cache_path)
        RTs = []
        for fid in range(start_frame, start_frame + self.seq_len):
            if self.load_meta:
                RT = torch.from_numpy(
                    np.array(meta["camera"]["RT_34flattened"][fid - 3]).reshape(3, 4)
                )
                RTs.append(RT)
            img = self.get_file(tp,
                                index,
                                self.File.RGB,
                                frame_id=fid).to(torch.float)[:3] / 255.0
            imgs.append(img)
            if self.load_mask:
                masks.append(
                    self.get_file(tp, index, self.File.MASK, frame_id=fid)
                )
            if self.load_fwd_flow:
                fwd_flows.append(
                    self.get_file(tp, index, self.File.FWD_FLOW, frame_id=fid)
                )
            if self.load_bwd_flow:
                bwd_flows.append(
                    -self.get_file(tp, index, self.File.BWD_FLOW, frame_id=fid)
                )
        imgs = torch.stack(imgs)
        r = (tp, imgs,)
        if self.load_mask:
            r = (*r, torch.stack(masks))
        if self.load_fwd_flow:
            r = (*r, torch.stack(fwd_flows))
        if self.load_bwd_flow:
            r = (*r, torch.stack(bwd_flows))

        if self.transforms is not None:
            r = self.transforms(r)
        if self.load_meta:
            return r, torch.stack(RTs)
        return r, None


class FlatCLEVRMOV(CLEVRMOV):
    def __init__(self, *args, min_frame=0, max_frame=None, filter=False, **kwargs):
        super().__init__(*args, **kwargs)
        max_frame = max_frame or self.seq_len
        self.index = []
        for i in range(super().__len__()):
            i = index_with_bias_and_limit(i, self.bias, self.limit)
            tp, index = self.inds[i]
            num_frames = min(self._num_frames(index), max_frame + 1)
            iids = [(tp, index, fid) for fid in range(3 + min_frame, num_frames + 1 - self.seq_len, self.seq_len)]
            iids = iids[9:] + iids[:9]  # Start with frame 10
            self.index.append(iids)
        # The following line flattens the index, such that it goes [Seq 1 f 1, Seq 2 f 1, ... Seq n f 1, Seq 1 f 2, ...]
        # This only matter for validation
        self.index = list(itertools.chain.from_iterable(zip(*self.index)))

        if filter:
            raise NotImplementedError("Filtering not implemented for FlatCLEVRMOV")

        self.seq_fid_to_idx_map = {
            (tp.name.replace('.tar', ''), fid): i for i, (tp, _, fid) in enumerate(self.index)
        }


    def __len__(self):
        return len(self.index)

    def idx_2_tarpath_index_frame_id(self, idx):
        return self.index[idx]



