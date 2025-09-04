from typing import Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from lib.action_head import fan_in_linear
from lib.normalize_ewma import NormalizeEwma


class ScaledMSEHead(nn.Module):
    """
    Linear output layer that scales itself so that targets are always normalized to N(0, 1)
    """

    def __init__(
        self, input_size: int, output_size: int, norm_type: Optional[str] = "ewma", norm_kwargs: Optional[Dict] = None
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.norm_type = norm_type

        self.linear = nn.Linear(self.input_size, self.output_size)

        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.normalizer = NormalizeEwma(output_size, **norm_kwargs)

    def reset_parameters(self):
        init.orthogonal_(self.linear.weight)
        fan_in_linear(self.linear)
        self.normalizer.reset_parameters()

    def forward(self, input_data):
        return self.linear(input_data)

    def loss(self, prediction, target):
        """
        Calculate the MSE loss between output and a target.
        'Prediction' has to be normalized while target is denormalized.
        Loss is calculated in a 'normalized' space.
        """
        return F.mse_loss(prediction, self.normalizer(target), reduction="mean")

    def denormalize(self, input_data):
        """Convert input value from a normalized space into the original one"""
        return self.normalizer.denormalize(input_data)

    def normalize(self, input_data):
        return self.normalizer(input_data)
