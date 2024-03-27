import functools
import itertools
import math
import os
import pickle
import re
import subprocess
import tempfile
from contextlib import contextmanager
from hashlib import md5, sha1

import numpy as np
import torch as th
import torch.distributed as dist
import torch.distributions as dis
import torch.nn.functional as F
from torch import nn

import lib.tree_util as tree_util
from lib import misc


def contextmanager_to_decorator(cm):
    def decorator(fn):
        @functools.wraps(fn)
        def newfn(*args, **kwargs):
            with cm():
                return fn(*args, **kwargs)

        return newfn

    return decorator


def have_cuda():
    return th.has_cuda


def default_device_type():
    return "cuda" if have_cuda() else "cpu"


no_grad = contextmanager_to_decorator(th.no_grad)
DEFAULT_DEVICE = th.device(type=default_device_type())


def set_default_torch_device(device):
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = th.device(device)


def dev():
    return DEFAULT_DEVICE


def zeros(*args, **kwargs):
    return th.zeros(*args, **kwargs, device=dev())


def ones(*args, **kwargs):
    return th.ones(*args, **kwargs, device=dev())


def arange(*args, **kwargs):
    return th.arange(*args, **kwargs, device=dev())


def NormedLinear(*args, scale=1.0, dtype=th.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    dtype = parse_dtype(dtype)
    if dtype == th.float32:
        out = nn.Linear(*args, **kwargs)
    elif dtype == th.float16:
        out = LinearF16(*args, **kwargs)
    else:
        raise ValueError(dtype)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


class LinearF16(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.half(), self.bias.half() if self.bias is not None else None)


class LayerNormF16(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight.half(), self.bias.half(), self.eps)


def LayerNorm(*args, dtype=th.float32, **kwargs):
    dtype = parse_dtype(dtype)
    if dtype == th.float32:
        out = nn.LayerNorm(*args, **kwargs)
    elif dtype == th.float16:
        out = LayerNormF16(*args, **kwargs)
    else:
        raise ValueError(dtype)
    out.weight.no_scale = True
    return out


def flatten_image(x):
    """
    Flattens last three dims
    """
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None, use_checkpoint=False):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x


@no_grad
def load_average_with_metadata(paths, overrides):
    n_models = len(paths)
    model, metadata = load_with_metadata(paths[0], overrides=overrides)
    for p in model.parameters():
        p.mul_(1 / n_models)
    for p in paths[1:]:
        new_model, _ = load_with_metadata(p, overrides=overrides)
        for (n1, p1), (n2, p2) in misc.safezip(model.named_parameters(), new_model.named_parameters()):
            assert n1 == n2, f"names {n1} and {n2} don't match"
            p1.add_(p2.mul_(1 / n_models))
    return model, metadata


def save_kwargs(fn):
    """
    This decorator passes through the user-provided kwargs and adds one more, called
    save_kwargs, mapping to {"create_fn" : name_of_decorated_fn, "kwargs" : other_kwargs}

    You put on this decorator on a function that creates a pytorch module. This will
    save the kwargs and the function that was used to create the module.
    This lets us restore the model state later.
    """

    @functools.wraps(fn)
    def wrapper(**kwargs):
        if "save_kwargs" in kwargs:
            return fn(**kwargs)
        else:
            sk = {**kwargs, "create_fn": f"{fn.__module__}:{fn.__name__}"}
            return fn(save_kwargs=sk, **kwargs)

    return wrapper


def parse_dtype(x):
    if isinstance(x, th.dtype):
        return x
    elif isinstance(x, str):
        if x == "float32" or x == "float":
            return th.float32
        elif x == "float64" or x == "double":
            return th.float64
        elif x == "float16" or x == "half":
            return th.float16
        elif x == "uint8":
            return th.uint8
        elif x == "int8":
            return th.int8
        elif x == "int16" or x == "short":
            return th.int16
        elif x == "int32" or x == "int":
            return th.int32
        elif x == "int64" or x == "long":
            return th.int64
        elif x == "bool":
            return th.bool
        else:
            raise ValueError(f"cannot parse {x} as a dtype")
    else:
        raise TypeError(f"cannot parse {type(x)} as dtype")


def index(x, i):
    """
    Batched, broadcasting index of x along dimension i.ndim.

    For example, if x has shape (1, 2, 3, 4, 5) and i has shape (1, 1, 3)
    then the result has shape (1, 2, 3, 5) and each value in i must be between 0 and 3.
    """
    assert x.ndim >= i.ndim + 1
    gather_dim = i.ndim
    while i.ndim < x.ndim:
        i = i.unsqueeze(-1)
    expand_shape = list(x.shape)
    expand_shape[gather_dim] = 1
    i = i.expand(*expand_shape)
    xi = th.gather(x, gather_dim, i)
    assert xi.shape[gather_dim] == 1
    return xi.squeeze(gather_dim)
