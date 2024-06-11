import math
from copy import deepcopy
from typing import Dict, List, Optional

from torch import nn
from torch.nn import functional as F

from lib import misc
from lib import torch_util as tu
from lib.util import FanInInitReLULayer


class CnnBasicBlock(nn.Module):
    """
    Residual basic block, as in ImpalaCNN. Preserves channel number and shape
    :param inchan: number of input channels
    :param init_scale: weight init scale multiplier
    """

    def __init__(
        self,
        inchan: int,
        init_scale: float = 1,
        log_scope="",
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        s = math.sqrt(init_scale)
        self.conv0 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv0",
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv1",
            **init_norm_kwargs,
        )

    def forward(self, x):
        x = x + self.conv1(self.conv0(x))
        return x


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN.
    :param inchan: number of input channels
    :param nblock: number of residual blocks after downsampling
    :param outchan: number of output channels
    :param init_scale: weight init scale multiplier
    :param pool: if true, downsample with max pool
    :param post_pool_groups: if not None, normalize with group norm with this many groups
    :param kwargs: remaining kwargs are passed into the blocks and layers
    """

    name = "Impala_CnnDownStack"

    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        log_scope: str = "",
        init_norm_kwargs: Dict = {},
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            log_scope=f"{log_scope}/firstconv",
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    log_scope=f"{log_scope}/block{i}",
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for i in range(nblock)
            ]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)
        x = tu.sequential(self.blocks, x, diag_name=self.name)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    """
    :param inshape: input image shape (height, width, channels)
    :param chans: number of residual downsample stacks. Each element is the number of
        filters per convolution in the stack
    :param outsize: output hidden size
    :param nblock: number of residual blocks per stack. Each block has 2 convs and a residual
    :param init_norm_kwargs: arguments to be passed to convolutional layers. Options can be found
        in ypt.model.util:FanInInitReLULayer
    :param dense_init_norm_kwargs: arguments to be passed to convolutional layers. Options can be found
        in ypt.model.util:FanInInitReLULayer
    :param kwargs: remaining kwargs are passed into the CnnDownStacks
    """

    name = "ImpalaCNN"

    def __init__(
        self,
        inshape: List[int],
        chans: List[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        h, w, c = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=math.sqrt(len(chans)),
                log_scope=f"downstack{i}",
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        self.dense = FanInInitReLULayer(
            misc.intprod(curshape),
            outsize,
            layer_type="linear",
            log_scope="imapala_final_dense",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )
        self.outsize = outsize

    def forward(self, x):
        b, t = x.shape[:-3]
        x = x.reshape(b * t, *x.shape[-3:])
        x = misc.transpose(x, "bhwc", "bchw")
        x = tu.sequential(self.stacks, x, diag_name=self.name)
        x = x.reshape(b, t, *x.shape[1:])
        x = tu.flatten_image(x)
        x = self.dense(x)
        return x
