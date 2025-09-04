from copy import deepcopy
from email import policy
from typing import Dict, Optional

import numpy as np
import torch as th
from gym3.types import DictType
from torch import nn
from torch.nn import functional as F

from lib.action_head import make_action_head
from lib.action_mapping import CameraHierarchicalMapping
from lib.impala_cnn import ImpalaCNN
from lib.normalize_ewma import NormalizeEwma
from lib.scaled_mse_head import ScaledMSEHead
from lib.tree_util import tree_map
from lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from lib.misc import transpose


class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(th.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(th.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        x = img.to(dtype=th.float32)
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        return self.linear(self.cnn(img))


class MinecraftPolicy(nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(
        self,
        recurrence_type="lstm",
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256,
        hidsize=512,
        single_output=False,  # True if we don't need separate outputs for action/value outputs
        img_shape=None,
        scale_input_img=True,
        only_img_input=False,
        init_norm_kwargs={},
        impala_kwargs={},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,  # Not needed for transformer
        **unused_kwargs,
    ):
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}

        self.single_output = single_output

        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type

        self.recurrent_layer = None
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = th.nn.LayerNorm(hidsize)

    def output_latent_size(self):
        return self.hidsize

    def forward(self, ob, state_in, context):
        first = context["first"]

        x = self.img_preprocess(ob["img"])
        x = self.img_process(x)

        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_out
        return (pi_latent, vf_latent), state_out

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None


class MinecraftAgentPolicy(nn.Module):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs):
        super().__init__()
        self.net = MinecraftPolicy(**policy_kwargs)

        self.action_space = action_space

        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
        return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)

    def make_action_head(self, pi_out_size: int, **pi_head_opts):
        return make_action_head(self.action_space, pi_out_size, **pi_head_opts)

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    def forward(self, obs, first: th.Tensor, state_in):
        if isinstance(obs, dict):
            # We don't want to mutate the obs input.
            obs = obs.copy()

            # If special "mask" key is in obs,
            # It's for masking the logits.
            # We take it out (the network doesn't need it)
            mask = obs.pop("mask", None)
        else:
            mask = None

        (pi_h, v_h), state_out = self.net(obs, state_in, context={"first": first})

        pi_logits = self.pi_head(pi_h, mask=mask)
        vpred = self.value_head(v_h)

        return (pi_logits, vpred, None), state_out

    def get_logprob_of_action(self, pd, action):
        """
        Get logprob of taking action `action` given probability distribution
        (see `get_gradient_for_action` to get this distribution)
        """
        ac = tree_map(lambda x: x.unsqueeze(1), action)
        log_prob = self.pi_head.logprob(ac, pd)
        assert not th.isnan(log_prob).any()
        return log_prob[:, 0]

    def get_kl_of_action_dists(self, pd1, pd2):
        """
        Get the KL divergence between two action probability distributions
        """
        return self.pi_head.kl_divergence(pd1, pd2)

    def get_output_for_observation(self, obs, state_in, first):
        """
        Return gradient-enabled outputs for given observation.

        Use `get_logprob_of_action` to get log probability of action
        with the given probability distribution.

        Returns:
          - probability distribution given observation
          - value prediction for given observation
          - new state
        """
        # We need to add a fictitious time dimension everywhere
        obs = tree_map(lambda x: x.unsqueeze(1), obs)
        first = first.unsqueeze(1)

        (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in)

        return pd, self.value_head.denormalize(vpred)[:, 0], state_out

    @th.no_grad()
    def act(self, obs, first, state_in, stochastic: bool = True, taken_action=None, return_pd=False):
        # We need to add a fictitious time dimension everywhere
        obs = tree_map(lambda x: x.unsqueeze(1), obs)
        first = first.unsqueeze(1)

        (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in)

        if taken_action is None:
            ac = self.pi_head.sample(pd, deterministic=not stochastic)
        else:
            ac = tree_map(lambda x: x.unsqueeze(1), taken_action)
        log_prob = self.pi_head.logprob(ac, pd)
        assert not th.isnan(log_prob).any()

        # After unsqueezing, squeeze back to remove fictitious time dimension
        result = {"log_prob": log_prob[:, 0], "vpred": self.value_head.denormalize(vpred)[:, 0]}
        if return_pd:
            result["pd"] = tree_map(lambda x: x[:, 0], pd)
        ac = tree_map(lambda x: x[:, 0], ac)

        return ac, state_out, result

    @th.no_grad()
    def v(self, obs, first, state_in):
        """Predict value for a given mdp observation"""
        obs = tree_map(lambda x: x.unsqueeze(1), obs)
        first = first.unsqueeze(1)

        (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in)

        # After unsqueezing, squeeze back
        return self.value_head.denormalize(vpred)[:, 0]


class InverseActionNet(MinecraftPolicy):
    """
    Args:
        conv3d_params: PRE impala 3D CNN params. They are just passed into th.nn.Conv3D.
    """

    def __init__(
        self,
        hidsize=512,
        conv3d_params=None,
        **MCPoliy_kwargs,
    ):
        super().__init__(
            hidsize=hidsize,
            # If we're using 3dconv, then we normalize entire impala otherwise don't
            # normalize the first impala layer since we normalize the input
            first_conv_norm=conv3d_params is not None,
            **MCPoliy_kwargs,
        )
        self.conv3d_layer = None
        if conv3d_params is not None:
            # 3D conv is the first layer, so don't normalize its input
            conv3d_init_params = deepcopy(self.init_norm_kwargs)
            conv3d_init_params["group_norm_groups"] = None
            conv3d_init_params["batch_norm"] = False
            self.conv3d_layer = FanInInitReLULayer(
                layer_type="conv3d",
                log_scope="3d_conv",
                **conv3d_params,
                **conv3d_init_params,
            )

    def forward(self, ob, state_in, context):
        first = context["first"]
        x = self.img_preprocess(ob["img"])

        # Conv3D Prior to Impala
        if self.conv3d_layer is not None:
            x = self._conv3d_forward(x)

        # Impala Stack
        x = self.img_process(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)

        x = F.relu(x, inplace=False)

        pi_latent = self.lastlayer(x)
        pi_latent = self.final_ln(x)
        return (pi_latent, None), state_out

    def _conv3d_forward(self, x):
        # Convert from (B, T, H, W, C) -> (B, H, W, C, T)
        x = transpose(x, "bthwc", "bcthw")
        new_x = []
        for mini_batch in th.split(x, 1):
            new_x.append(self.conv3d_layer(mini_batch))
        x = th.cat(new_x)
        # Convert back
        x = transpose(x, "bcthw", "bthwc")
        return x


class InverseActionPolicy(nn.Module):
    def __init__(
        self,
        action_space,
        pi_head_kwargs=None,
        idm_net_kwargs=None,
    ):
        super().__init__()
        self.action_space = action_space

        self.net = InverseActionNet(**idm_net_kwargs)

        pi_out_size = self.net.output_latent_size()

        pi_head_kwargs = {} if pi_head_kwargs is None else pi_head_kwargs

        self.pi_head = self.make_action_head(pi_out_size=pi_out_size, **pi_head_kwargs)

    def make_action_head(self, **kwargs):
        return make_action_head(self.action_space, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()

    def forward(self, obs, first: th.Tensor, state_in, **kwargs):
        if isinstance(obs, dict):
            # We don't want to mutate the obs input.
            obs = obs.copy()

            # If special "mask" key is in obs,
            # It's for masking the logits.
            # We take it out (the network doesn't need it)
            mask = obs.pop("mask", None)
        else:
            mask = None

        (pi_h, _), state_out = self.net(obs, state_in=state_in, context={"first": first}, **kwargs)
        pi_logits = self.pi_head(pi_h, mask=mask)
        return (pi_logits, None, None), state_out

    @th.no_grad()
    def predict(
        self,
        obs,
        deterministic: bool = True,
        **kwargs,
    ):
        (pd, _, _), state_out = self(obs=obs, **kwargs)

        ac = self.pi_head.sample(pd, deterministic=deterministic)
        log_prob = self.pi_head.logprob(ac, pd)

        assert not th.isnan(log_prob).any()

        result = {"log_prob": log_prob, "pd": pd}

        return ac, state_out, result

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)
