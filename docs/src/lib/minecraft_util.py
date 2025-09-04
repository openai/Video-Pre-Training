import functools
import inspect
from typing import Optional, Tuple

import numpy as np
import torch

from lib.action_head import (CategoricalActionHead, DiagGaussianActionHead,
                             DictActionHead)


def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def get_norm_entropy_from_cat_head(module, name, masks, logits):
    # Note that the mask has already been applied to the logits at this point
    entropy = -torch.sum(torch.exp(logits) * logits, dim=-1)
    if name in masks:
        n = torch.sum(masks[name], dim=-1, dtype=torch.float)
        norm_entropy = entropy / torch.log(n)
        # When the mask only allows one option the normalized entropy makes no sense
        # as it is basically both maximal (the distribution is as uniform as it can be)
        # and minimal (there is no variance at all).
        # A such, we ignore them for purpose of calculating entropy.
        zero = torch.zeros_like(norm_entropy)
        norm_entropy = torch.where(n.eq(1.0), zero, norm_entropy)
        count = n.not_equal(1.0).int()
    else:
        n = torch.tensor(logits.shape[-1], dtype=torch.float)
        norm_entropy = entropy / torch.log(n)
        count = torch.ones_like(norm_entropy, dtype=torch.int)

    # entropy is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
    for _ in module.output_shape[:-1]:
        norm_entropy = norm_entropy.sum(dim=-1)
        count = count.sum(dim=-1)
    return norm_entropy, count


def get_norm_cat_entropy(module, masks, logits, template) -> Tuple[torch.Tensor, torch.Tensor]:
    entropy_sum = torch.zeros_like(template, dtype=torch.float)
    counts = torch.zeros_like(template, dtype=torch.int)
    for k, subhead in module.items():
        if isinstance(subhead, DictActionHead):
            entropy, count = get_norm_cat_entropy(subhead, masks, logits[k], template)
        elif isinstance(subhead, CategoricalActionHead):
            entropy, count = get_norm_entropy_from_cat_head(subhead, k, masks, logits[k])
        else:
            continue
        entropy_sum += entropy
        counts += count
    return entropy_sum, counts


def get_diag_guassian_entropy(module, logits, template) -> Optional[torch.Tensor]:
    entropy_sum = torch.zeros_like(template, dtype=torch.float)
    count = torch.zeros(1, device=template.device, dtype=torch.int)
    for k, subhead in module.items():
        if isinstance(subhead, DictActionHead):
            entropy_sum += get_diag_guassian_entropy(subhead, logits[k], template)
        elif isinstance(subhead, DiagGaussianActionHead):
            entropy_sum += module.entropy(logits)
        else:
            continue
        count += 1
    return entropy_sum / count
