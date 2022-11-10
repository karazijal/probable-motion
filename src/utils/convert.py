import itertools

import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists or subclasses of object; found {}")

# Modified from torch souce code
def callate_with_objects(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return callate_with_objects([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: callate_with_objects([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(callate_with_objects(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [callate_with_objects(samples) for samples in transposed]
    else:
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def cast_like(maybe_tensor, example_tensor, **tspec):
    """
    Makes <maybe_tensor> be like <example_tensor> (dtype, device, ndim) etc...
    New dimensions are added at the end.
    """
    if not torch.is_tensor(maybe_tensor):
        maybe_tensor = torch.tensor(maybe_tensor)
    maybe_tensor = maybe_tensor.to(tspec.get('device', example_tensor.device)).to(
        tspec.get('dtype', example_tensor.dtype))
    shape = [*maybe_tensor.shape, *[1] * len(example_tensor.shape)]
    if shape:
        maybe_tensor = maybe_tensor.view(*shape)
    return maybe_tensor



def ensure_non_scalar(tensor):
    if not len(tensor.shape):
        tensor = tensor.view(1)
    return tensor


def list_of_dicts_2_dict_of_lists(list_of_dicts):
    keys = set(itertools.chain.from_iterable(list_of_dicts))
    out_dict = {}
    for k in keys:
        out_dict[k] = [d[k] for d in list_of_dicts if k in d]
    return out_dict


def list_of_dicts_2_dict_of_tensors(list_of_dicts, **tspec):
    d = callate_with_objects(list_of_dicts)
    if tspec:
        for k in d:
            if torch.is_tensor(d[k]):
                for v in tspec.values():
                    d[k] = d[k].to(v)
    return d

def to_batchxtime(dict_of_tensors):
    r = {}
    for k in dict_of_tensors:
        if torch.is_tensor(dict_of_tensors[k]) and len(dict_of_tensors[k].shape) >= 2:
            b, t, *rest = dict_of_tensors[k].shape
            r[k] = dict_of_tensors[k].reshape(b*t, *rest)
        else:
            r[k] = dict_of_tensors[k]
    return r

def to_batch_and_time(dict_of_tensors, t=1):
    r = {}
    for k in dict_of_tensors:
        if torch.is_tensor(dict_of_tensors[k]) and len(dict_of_tensors[k].shape) >= 2:
            b, *rest = dict_of_tensors[k].shape
            r[k] = dict_of_tensors[k].reshape(b * t, *rest)
        else:
            r[k] = dict_of_tensors[k]
            # r[k] = [dict_of_tensors[k][i:i+t] for i in range(0, len(dict_of_tensors[k]), t)]
    return r

def to_5dim_hard_mask(mask, K=None, device=None, dtype=None):
    ts = {'device':device or mask.device}
    if mask.dtype in {torch.float16, torch.float32, torch.float64, torch.half, torch.float, torch.double}:
        ts['dtype'] = dtype or mask.dtype
    else:
        ts['dtype'] = dtype or torch.get_default_dtype()  # use default float type

    shape = mask.shape
    if len(shape) == 2:
        h, w = shape
        b = 1
        cs = []
        k = None
    elif len(shape) == 3:
        b, h, w = shape
        cs = []
        k = None
    elif len(shape) == 4:
        b, c_or_k, h, w = shape
        if c_or_k == 1:
            cs = [c_or_k]
            k = None
        else:
            cs = []
            k = c_or_k
    elif len(shape) >= 5:
        b, k, *cs, h, w = shape
    else:
        raise ValueError(f"Mask should be at least 2D, got {shape}")
    if any(c != 1 for c in cs):
        raise ValueError(f"Unknown mask shape {shape}. Should be of form [b', k', 1*, h, w]. dim' -- optional.")

    K = K or k or mask.max().item() + 1
    if k is not None:
        mask = mask.view(b, k, 1, h, w).argmax(1, keepdim=True)
    else:
        mask = mask.view(b, 1, 1, h, w)
    mask = mask.to(ts['dtype']).to(ts['device'])
    mask = (mask == torch.arange(K, **ts).view(1, -1, 1, 1, 1)).to(ts['dtype'])
    return mask

def sort_by_sum(tensor, sort_dim, sum_dims=-1, descending=False):
    scores = tensor.sum(dim=sum_dims, keepdim=True)
    indexes = scores.argsort(sort_dim, descending)
    return torch.gather(tensor, sort_dim, indexes.expand_as(tensor))
