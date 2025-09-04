import functools

import torch as th
from torch import nn

import lib.xf as xf
from lib.minecraft_util import store_args
from lib.tree_util import tree_map


@functools.lru_cache()
def get_band_diagonal_mask(t: int, T: int, maxlen: int, batchsize: int, device: th.device) -> th.Tensor:
    """Returns a band diagonal mask which is causal (upper triangle is masked)
    and such that any frame can only view up to maxlen total past frames
    including the current frame.

    Example Masks: Here 0 means that frame is masked and we mask it by adding a huge number to the attention logits (see orc.xf)
        t = 3, T = 3, maxlen = 3
          T
        t 1 0 0 |  mask out T > t
          1 1 0 |
          1 1 1 |
        t = 3, T = 6, maxlen = 3
        t 0 1 1 1 0 0 |  mask out T > t
          0 0 1 1 1 0 |
          0 0 0 1 1 1 |

    Args:
        t: number of rows (presumably number of frames recieving gradient)
        T: number of cols (presumably t + past context that isn't being gradient updated)
        maxlen: maximum number of frames (including current frame) any frame can attend to
        batchsize: number of masks to return
        device: torch device to place mask on

    Returns:
        Boolean mask of shape (batchsize, t, T)
    """
    m = th.ones(t, T, dtype=bool)
    m.tril_(T - t)  # Mask out upper triangle
    if maxlen is not None and maxlen < T:  # Mask out lower triangle
        m.triu_(T - t - maxlen + 1)
    m_btT = m[None].repeat_interleave(batchsize, dim=0)
    m_btT = m_btT.to(device=device)
    return m_btT


def get_mask(first_b11: th.Tensor, state_mask: th.Tensor, t: int, T: int, maxlen: int, heads: int, device) -> th.Tensor:
    """Returns a band diagonal mask that respects masking past states (columns 0:T-t inclusive)
        if first_b11 is True. See get_band_diagonal_mask for how the base mask is computed.
        This function takes that mask and first zeros out any past context if first_b11 is True.

        Say our context is in chunks of length t (so here T = 4t). We see that in the second batch we recieved first=True
        context     t t t t
        first       F T F F
        Now, given this the mask should mask out anything prior to T < t; however since we don't have access to the past first_b11's
        we need to keep a state of the mask at those past timesteps. This is what state_mask is.

        In particular state_mask is a [b, t, T - t] mask matrix that contains the mask for the past T - t frames.

    Args: (See get_band_diagonal_mask for remaining args)
        first_b11: boolean tensor with shape [batchsize, 1, 1] indicating if the first timestep for each batch element had first=True
        state_mask: mask tensor of shape [b, t, T - t]
        t: number of mask rows (presumably number of frames for which we take gradient)
        T: number of mask columns (t + the number of past frames we keep in context)
        maxlen: actual context length
        heads: number of attention heads
        device: torch device

    Returns:
        m_btT: Boolean mask of shape (batchsize * heads, t, T)
        state_mask: updated state_mask
    """
    b = first_b11.shape[0]

    if state_mask is None:
        state_mask = th.zeros((b, 1, T - t), dtype=bool, device=device)

    m_btT = get_band_diagonal_mask(t, T, maxlen, b, device).clone()  # Should be shape B, t, T
    not_first = ~first_b11.to(device=device)
    m_btT[:, :, :-t] &= not_first  # Zero out anything in the past if first is true
    m_btT[:, :, :-t] &= state_mask
    m_bhtT = m_btT[:, None].repeat_interleave(heads, dim=1)
    m_btT = m_bhtT.reshape((b * heads), t, T)

    # Update state_mask such that it reflects the most recent first
    state_mask = th.cat(
        [
            state_mask[:, :, t:] & not_first,
            th.ones((b, 1, min(t, T - t)), dtype=bool, device=device),
        ],
        dim=-1,
    )

    return m_btT, state_mask


class MaskedAttention(nn.Module):
    """
    Transformer self-attention layer that removes frames from previous episodes from the hidden state under certain constraints.

    The constraints are:
    - The "first" flag can only be true for the first timestep of each batch. An assert will fire if other timesteps have first = True.

    input_size: The dimension of the input (which also happens to be the size of the output)
    memory_size: The number of frames to keep in the inner state. Note that when attending, we will be able to attend
                 to both the frames in the inner state (which presumably won't have gradients anymore) and the frames
                 in the batch. "mask" for some additional considerations on this.
    heads: The number of attention heads to use. Note that we will split the input into this number of heads, so
           input_size needs to be divisible by heads.
    timesteps: number of timesteps with which we'll be taking gradient
    mask: Can be "none" or "clipped_causal". "clipped_causal" is a normal causal mask but solves the following minor problem:
        if you have a state of length 128 and a batch of 128 frames, then the first frame of your batch will be able to
        attend to 128 previous frames, but the last one will be able to attend to 255 previous frames. In this example,
        "clipped_causal" will make it so that the last frame can only attend to 128 previous frames, so that there is no
        bias coming from the position in the batch. None simply allows you to attend to any frame in the state + batch,
        which means you can also attend to future frames.
    """

    @store_args
    def __init__(
        self,
        input_size,
        memory_size: int,
        heads: int,
        timesteps: int,
        mask: str = "clipped_causal",
        init_scale=1,
        norm="none",
        log_scope="sa",
        use_muP_factor=False,
    ):
        super().__init__()

        assert mask in {"none", "clipped_causal"}
        assert memory_size >= 0

        self.maxlen = memory_size - timesteps
        if mask == "none":
            mask = None

        self.orc_attn = xf.All2All(heads, self.maxlen, mask=mask is not None)
        self.orc_block = xf.SelfAttentionLayer(
            input_size,
            self.orc_attn,
            scale=init_scale,
            relattn=True,
            cache_keep_len=self.maxlen,
            norm=norm,
            log_scope=log_scope,
            use_muP_factor=use_muP_factor,
        )

    def initial_state(self, batchsize: int, device=None):
        """Return the initial state mask (None) and the initial state of the transformer (zerod out keys and queries)"""
        state = self.orc_block.initial_state(batchsize, initial_T=self.maxlen)
        state_mask = None
        if device is not None:
            state = tree_map(lambda x: x.to(device), state)
        return state_mask, state

    def forward(self, input_bte, first_bt, state):
        """Forward propagation of a single layer"""
        state_mask, xf_state = state
        t = first_bt.shape[1]
        if self.mask == "clipped_causal":
            new_mask, state_mask = get_mask(
                first_b11=first_bt[:, [[0]]],
                state_mask=state_mask,
                t=t,
                T=t + self.maxlen,
                maxlen=self.maxlen,
                heads=self.heads,
                device=input_bte.device,
            )
            self.orc_block.attn.mask = new_mask
        output, xf_state = self.orc_block(input_bte, xf_state)

        return output, (state_mask, xf_state)

    def get_log_keys(self):
        # These are logged in xf.SelfAttentionLayer
        return [f"activation_{stat}/{self.log_scope}/{k}" for k in ["K", "Q", "V", "A", "Aproj"] for stat in ["mean", "std"]]
