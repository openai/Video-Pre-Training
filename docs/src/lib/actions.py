import attr
import minerl.herobraine.hero.mc as mc
import numpy as np

from lib.minecraft_util import store_args


class Buttons:
    ATTACK = "attack"
    BACK = "back"
    FORWARD = "forward"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    SNEAK = "sneak"
    SPRINT = "sprint"
    USE = "use"
    DROP = "drop"
    INVENTORY = "inventory"

    ALL = [
        ATTACK,
        BACK,
        FORWARD,
        JUMP,
        LEFT,
        RIGHT,
        SNEAK,
        SPRINT,
        USE,
        DROP,
        INVENTORY,
    ] + [f"hotbar.{i}" for i in range(1, 10)]


class SyntheticButtons:
    # Composite / scripted actions
    CHANNEL_ATTACK = "channel-attack"

    ALL = [CHANNEL_ATTACK]


class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


class ActionTransformer:
    """Transforms actions between internal array and minerl env format."""

    @store_args
    def __init__(
        self,
        camera_maxval=10,
        camera_binsize=2,
        camera_quantization_scheme="linear",
        camera_mu=5,
    ):
        self.quantizer = CameraQuantizer(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            quantization_scheme=camera_quantization_scheme,
            mu=camera_mu,
        )

    def camera_zero_bin(self):
        return self.camera_maxval // self.camera_binsize

    def discretize_camera(self, xy):
        return self.quantizer.discretize(xy)

    def undiscretize_camera(self, pq):
        return self.quantizer.undiscretize(pq)

    def item_embed_id_to_name(self, item_id):
        return mc.MINERL_ITEM_MAP[item_id]

    def dict_to_numpy(self, acs):
        """
        Env format to policy output format.
        """
        act = {
            "buttons": np.stack([acs.get(k, 0) for k in Buttons.ALL], axis=-1),
            "camera": self.discretize_camera(acs["camera"]),
        }
        if not self.human_spaces:
            act.update(
                {
                    "synthetic_buttons": np.stack([acs[k] for k in SyntheticButtons.ALL], axis=-1),
                    "place": self.item_embed_name_to_id(acs["place"]),
                    "equip": self.item_embed_name_to_id(acs["equip"]),
                    "craft": self.item_embed_name_to_id(acs["craft"]),
                }
            )
        return act

    def numpy_to_dict(self, acs):
        """
        Numpy policy output to env-compatible format.
        """
        assert acs["buttons"].shape[-1] == len(
            Buttons.ALL
        ), f"Mismatched actions: {acs}; expected {len(Buttons.ALL)}:\n(  {Buttons.ALL})"
        out = {name: acs["buttons"][..., i] for (i, name) in enumerate(Buttons.ALL)}

        out["camera"] = self.undiscretize_camera(acs["camera"])

        return out

    def policy2env(self, acs):
        acs = self.numpy_to_dict(acs)
        return acs

    def env2policy(self, acs):
        nbatch = acs["camera"].shape[0]
        dummy = np.zeros((nbatch,))
        out = {
            "camera": self.discretize_camera(acs["camera"]),
            "buttons": np.stack([acs.get(k, dummy) for k in Buttons.ALL], axis=-1),
        }
        return out
