"""
Implementation of transformer and reshaping-based sparse transformer
"""
import functools
import math

import torch as th
from torch import nn
from torch.nn import functional as F

from lib import misc, mlp
from lib import torch_util as tu
from lib import util

SENTINEL = 0.1337


def attention(
    Q_bte,
    K_bTe,
    V_bTe,
    dtype,
    mask=True,
    extra_btT=None,
    maxlen=None,
    check_sentinel=False,
    use_muP_factor=False,
):
    """
    performs softmax(Q*K)*V operation

    t : output (write) time axis, possibly size=1 for just the last timestep
    T : input (read) time axis
    t < T is OK

    'check_sentinel' is used when you want to make it impossible to attend to certain keys.
    All keys where every value is equal to the constant SENTINEL will be ignored.
    Currently this is only used by StridedAttn.
    """
    assert Q_bte.dtype == K_bTe.dtype == dtype, f"{Q_bte.dtype}, {K_bTe.dtype}, {dtype} must all match"
    e = Q_bte.shape[2]
    if check_sentinel:
        invalid = (K_bTe == SENTINEL).int().sum(dim=-1) == e
        invalid = misc.reshape(invalid, "b, T", "b, 1, T")
    if isinstance(mask, th.Tensor):
        bias = (~mask).float() * -1e9
    elif mask:
        bias = get_attn_bias_cached(Q_bte.shape[1], K_bTe.shape[1], maxlen=maxlen, device=Q_bte.device, dtype=th.float32)
    else:
        bias = Q_bte.new_zeros((), dtype=th.float32)
    if extra_btT is not None:
        bias = bias + extra_btT
    # Equivalent to bias + (1 / math.sqrt(e)) * th.einsum("bte,bpe->btp", Q_bte, K_bte)
    # but faster:
    logit_btT = th.baddbmm(
        bias,
        Q_bte.float(),
        K_bTe.float().transpose(-1, -2),
        alpha=(1 / e) if use_muP_factor else (1 / math.sqrt(e)),
    )
    if check_sentinel:
        logit_btT = logit_btT - 1e9 * invalid.float()
    W_btT = th.softmax(logit_btT, dim=2).to(dtype)
    if callable(V_bTe):
        # This is used by the sharded video model to defer waiting on
        # the broadcast of the values until they're needed
        V_bTe = V_bTe()
    # th.einsum only lets you use lowercase letters, so 'p' for 'past'
    # means 'T'
    A_bte = th.einsum("btp,bpe->bte", W_btT, V_bTe)
    return A_bte


class Attn:
    """
    Defines an attention mechanism
    All the mechanisms here can be defined by two operations:
    1. preprocessing Q,K,V,R[=relative attention query]
        to move axes from embedding dimension to
        batch dimension, and possibly doing shifts.
    2. postprocessing the final result to move axes back to embedding
        axis.
    """

    def __init__(self, mask, maxlen):
        self.mask = mask
        self.maxlen = maxlen

    def preproc_qkv(self, Q_bte, K_bte, V_bte):
        raise NotImplementedError

    def preproc_r(self, R_btn):
        raise NotImplementedError


def split_heads(x_bte, h):
    b, t, e = x_bte.shape
    assert e % h == 0, "Embsize must be divisible by number of heads"
    q = e // h
    x_bthq = x_bte.reshape((b, t, h, q))
    x_bhtq = misc.transpose(x_bthq, "bthq", "bhtq")
    x_Btq = x_bhtq.reshape((b * h, t, q))
    return x_Btq


class All2All(Attn):
    def __init__(self, nhead, maxlen, mask=True, head_dim=None):
        super().__init__(mask=mask, maxlen=maxlen)
        assert (nhead is None) != (head_dim is None), "exactly one of nhead and head_dim must be specified"
        self.h = nhead
        self.head_dim = head_dim

    def preproc_qkv(self, *xs):
        q = xs[0].shape[-1]
        for x in xs:
            assert x.shape[-1] == q, "embedding dimensions do not match"
        h = self.h or misc.exact_div(q, self.head_dim)
        postproc = functools.partial(self.postproc_a, h=h)
        return (postproc, *tuple(split_heads(x, h) for x in xs))

    def preproc_r(self, R_btn):
        _, ret = self.preproc_qkv(R_btn)
        return ret

    def postproc_a(self, A_Btq, h):
        B, t, q = A_Btq.shape
        b = B // h
        A_bhtq = A_Btq.reshape((b, h, t, q))
        A_bthq = misc.transpose(A_bhtq, "bhtq", "bthq")
        A_bte = A_bthq.reshape((b, t, h * q))
        return A_bte


def _required_padding(dim, target_div):
    if dim % target_div == 0:
        return 0
    else:
        return target_div - dim % target_div


class StridedAttn(Attn):
    def __init__(self, nhead, stride, maxlen, mask=True):
        super().__init__(mask=mask, maxlen=maxlen)
        self.h = nhead
        self.stride = stride

    def _preproc(self, x, name, Q_t=None, Q_pad=None):
        x, undo = misc.reshape_undo(x, "b, t*stride, e", "b, 1, t, stride*e", stride=self.stride)
        if name == "Q":
            Q_pad = _required_padding(x.shape[2], self.maxlen)
        original_t = x.shape[2]
        x = F.pad(x, (0, 0, 0, Q_pad), value=SENTINEL)
        undo = misc.compose_undo(undo, lambda x: x[:, :, :original_t])
        if name == "Q":
            Q_t = x.shape[2]
            assert Q_t % self.maxlen == 0, f"{Q_t} % {self.maxlen} != 0"
        else:
            required_len = Q_t + self.maxlen
            if x.shape[2] < required_len:
                x = F.pad(x, (0, 0, required_len - x.shape[2], 0), value=SENTINEL)
            assert x.shape[2] >= required_len
            back = x[:, :, -Q_t - self.maxlen : -self.maxlen]
            front = x[:, :, -Q_t:]
            x = th.cat([back, front], dim=1)
        _, _, t, _ = x.shape
        assert t == Q_t, f"{t} != {Q_t}"
        x, undo = misc.reshape_undo(
            x,
            "b, pad_shift, t*maxlen, stride*h*q",
            "b, pad_shift, t, maxlen, stride, h, q",
            maxlen=self.maxlen,
            h=self.h,
            stride=self.stride,
            undo=undo,
        )
        x, undo = misc.transpose_undo(x, "bptmshq", "bthspmq", undo=undo)
        x, undo = misc.reshape_undo(
            x,
            "b, t, h, stride, pad_shift, maxlen, q",
            "b*t*h*stride, pad_shift*maxlen, q",
            undo=undo,
        )
        if name == "Q":
            return x, undo, Q_t, Q_pad
        else:
            return x

    def preproc_qkv(self, Q_bte, K_bte, V_bte):
        pad = _required_padding(Q_bte.shape[1], self.stride)
        if pad:
            Q_bte = F.pad(Q_bte, (0, 0, 0, pad), value=SENTINEL)
            K_bte = F.pad(K_bte, (0, 0, 0, pad), value=SENTINEL) if K_bte is not None else None
            V_bte = F.pad(V_bte, (0, 0, 0, pad), value=SENTINEL) if V_bte is not None else None
            undo = lambda x, pad=pad: x[:, :-pad]
        else:
            undo = None
        if K_bte is not None:
            pad = _required_padding(K_bte.shape[1], self.stride)
            if pad:
                K_bte = F.pad(K_bte, (0, 0, pad, 0), value=SENTINEL)
                V_bte = F.pad(V_bte, (0, 0, pad, 0), value=SENTINEL)
        assert Q_bte.shape[1] % self.stride == 0
        assert K_bte is None or K_bte.shape[1] % self.stride == 0
        assert V_bte is None or V_bte.shape[1] % self.stride == 0
        Q, postproc, Q_t, Q_pad = self._preproc(Q_bte, "Q")
        postproc = misc.compose_undo(undo, postproc)
        return (
            postproc,
            Q,
            self._preproc(K_bte, "K", Q_t=Q_t, Q_pad=Q_pad) if K_bte is not None else None,
            self._preproc(V_bte, "V", Q_t=Q_t, Q_pad=Q_pad) if V_bte is not None else None,
        )

    def preproc_r(self, R_bte):
        _, R, _, _ = self.preproc_qkv(R_bte, None, None)
        return R


Q_SCALE = 0.1
K_SCALE = 0.2
V_SCALE = 1.0
PROJ_SCALE = 1.0
MLP0_SCALE = 1.0
MLP1_SCALE = 1.0
R_SCALE = 0.1
B_SCALE = 0.2


class AttentionLayerBase(nn.Module):
    def __init__(
        self,
        *,
        attn,
        scale,
        x_size,
        c_size,
        qk_size,
        v_size,
        dtype,
        relattn=False,
        seqlens=None,
        separate=False,
    ):
        super().__init__()
        dtype = tu.parse_dtype(dtype)
        self.attn = attn
        self.x_size = x_size
        self.c_size = c_size
        s = math.sqrt(scale)
        separgs = dict(seqlens=seqlens, separate=separate)
        self.q_layer = MultiscaleLinear(x_size, qk_size, name="q", scale=Q_SCALE, dtype=dtype, **separgs)
        self.k_layer = MultiscaleLinear(c_size, qk_size, name="k", scale=K_SCALE, bias=False, dtype=dtype, **separgs)
        self.v_layer = MultiscaleLinear(c_size, v_size, name="v", scale=V_SCALE * s, bias=False, dtype=dtype, **separgs)
        self.proj_layer = MultiscaleLinear(v_size, x_size, name="proj", scale=PROJ_SCALE * s, dtype=dtype, **separgs)
        self.relattn = relattn
        maxlen = attn.maxlen
        assert maxlen > 0 or not attn.mask
        if self.relattn:
            nbasis = 10
            self.r_layer = tu.NormedLinear(x_size, nbasis * attn.h, scale=R_SCALE, dtype=dtype)
            self.b_nd = nn.Parameter(th.randn(nbasis, maxlen) * B_SCALE)
        self.maxlen = maxlen
        self.dtype = dtype

    def relattn_logits(self, X_bte, T):
        R_btn = self.r_layer(X_bte).float()
        R_btn = self.attn.preproc_r(R_btn)
        t = R_btn.shape[1]
        D_ntT = util.bandify(self.b_nd, t, T)
        extra_btT = th.einsum("btn,ntp->btp", R_btn, D_ntT)
        return extra_btT


def quick_gelu(x):
    return x * th.sigmoid(1.702 * x)


def act(actname, x):
    if actname == "relu":
        return F.relu(x)
    elif actname == "gelu":
        return quick_gelu(x)
    elif actname == "none":
        return x
    else:
        raise NotImplementedError(actname)


class SelfAttentionLayer(AttentionLayerBase):
    """
    Residual attention layer that takes a single tensor x and has it attend to itself
    Has the form
        output = x + f(x)
    """

    def __init__(
        self,
        x_size,
        attn,
        scale,
        dtype="float32",
        norm="layer",
        cache_keep_len=None,
        relattn=False,
        log_scope="sa",
        use_muP_factor=False,
        **kwargs,
    ):
        super().__init__(
            x_size=x_size,
            c_size=x_size,
            qk_size=x_size,
            v_size=x_size,
            attn=attn,
            scale=scale,
            relattn=relattn,
            dtype=dtype,
            **kwargs,
        )
        self.ln_x = util.get_norm(norm, x_size, dtype=dtype)
        if cache_keep_len is None:
            if hasattr(attn, "cache_keep_len"):
                cache_keep_len = attn.cache_keep_len
            else:
                if isinstance(attn, StridedAttn):
                    stride = attn.stride
                else:
                    stride = 1
                cache_keep_len = stride * attn.maxlen
        self.cache_keep_len = cache_keep_len
        self.log_scope = log_scope
        self.use_muP_factor = use_muP_factor

    def residual(self, X_bte, state):
        X_bte = self.ln_x(X_bte)
        Q_bte = self.q_layer(X_bte)
        K_bte = self.k_layer(X_bte)
        V_bte = self.v_layer(X_bte)
        if state:
            state, K_bte, V_bte = self.update_state(state, K_bte, V_bte)
        postproc_closure, Q_bte, K_bte, V_bte = self.attn.preproc_qkv(Q_bte, K_bte, V_bte)
        extra_btT = self.relattn_logits(X_bte, K_bte.shape[1]) if self.relattn else None
        A_bte = attention(
            Q_bte,
            K_bte,
            V_bte,
            mask=self.attn.mask,
            extra_btT=extra_btT,
            maxlen=self.maxlen,
            dtype=self.dtype,
            check_sentinel=isinstance(self.attn, StridedAttn),
            use_muP_factor=self.use_muP_factor,
        )
        A_bte = postproc_closure(A_bte)
        Aproj_bte = self.proj_layer(A_bte)
        return Aproj_bte, state

    def forward(self, X_bte, state):
        R_bte, state = self.residual(X_bte, state)
        return X_bte + R_bte, state

    def stateless_forward(self, X_bte):
        out_bte, _state = self.forward(X_bte, None)
        return out_bte

    def update_state(self, state, K_bte, V_bte):
        def append(prev, new):
            """
            Given `prev` keys from cache, and `new` keys,
            returns (cache, full), where
            - cache goes into the output state, length chosen so that on the
                next timestep, there are enough cached timesteps to get the full
                context of lenth self.maxlen.
            - full is used for the current forward pass, with length chosen so
                that the first timestep new[:, 0] gets to see a context of
                self.maxlen.
            """
            tprev = prev.shape[1]
            startfull = max(tprev - self.cache_keep_len, 0)
            full = th.cat([prev[:, startfull:], new], dim=1)
            outstate = full[:, max(full.shape[1] - (self.cache_keep_len), 0) :]
            # To see that the preceding slicing is correct, consider the case
            # that maxlen==1. Then `full` only consists of `new`, and
            # `outstate` is empty
            return outstate, full

        instate_K, instate_V = state
        outstate_K, K_bte = append(instate_K, K_bte)
        outstate_V, V_bte = append(instate_V, V_bte)
        assert outstate_K.shape[-2] <= self.cache_keep_len
        return (outstate_K, outstate_V), K_bte, V_bte

    def initial_state(self, batchsize, initial_T=0):
        return (
            tu.zeros((batchsize, initial_T, self.x_size), dtype=self.dtype),
            tu.zeros((batchsize, initial_T, self.x_size), dtype=self.dtype),
        )

    def empty_state(self):
        return None


class PointwiseLayer(nn.Module):
    """
    Residual MLP applied at each timestep
    """

    def __init__(self, x_size, scale, dtype, norm, actname="relu", mlp_ratio=2):
        super().__init__()
        s = math.sqrt(scale)
        self.ln = util.get_norm(norm, x_size, dtype=dtype)
        self.mlp = mlp.MLP(
            insize=x_size,
            nhidlayer=1,
            outsize=x_size,
            hidsize=int(x_size * mlp_ratio),
            hidactiv=functools.partial(act, actname),
            dtype=dtype,
        )
        self.mlp.layers[0].weight.data *= MLP0_SCALE * s
        self.mlp.layers[1].weight.data *= MLP1_SCALE * s

    def residual(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


def _is_separate(sep, name):
    if isinstance(sep, bool):
        return sep
    assert isinstance(sep, set)
    if name in sep:
        sep.remove(name)
        return True
    else:
        return False


def make_maybe_multiscale(make_fn, *args, seqlens, separate, name, **kwargs):
    """
    This function either creates one instance of a module or creates
    a separate instance of the module for each resolution of the image,
    determined by the `separate` parameter. We create separate modules
    if `separate` is True or if `separate` is a set containing `name`.
    """
    if _is_separate(separate, name):
        modules = [make_fn(*args, **kwargs) for _ in seqlens]
        return SplitCallJoin(modules, seqlens)
    else:
        return make_fn(*args, **kwargs)


class SplitCallJoin(nn.Module):
    def __init__(self, mods, seqlens):
        super().__init__()
        self.mods = nn.ModuleList(mods)
        self.seqlens = seqlens

    def forward(self, x):
        tl = sum(self.seqlens)
        x, undo = misc.reshape_undo(x, "..., z*tl, e", "..., z, tl, e", tl=tl)
        x = list(th.split(x, self.seqlens, dim=-2))
        new_x = []
        for x, mod in misc.safezip(x, self.mods):
            x, this_undo = misc.reshape_undo(x, "..., z, l, e", "..., z*l, e")
            x = mod(x)
            x = this_undo(x)
            new_x.append(x)
        x = th.cat(new_x, dim=-2)
        x = undo(x)
        return x


MultiscaleLinear = functools.partial(make_maybe_multiscale, tu.NormedLinear)
MultiscalePointwise = functools.partial(make_maybe_multiscale, PointwiseLayer)
