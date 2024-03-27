import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces

from lib.action_mapping import CameraHierarchicalMapping, IDMActionMapping
from lib.actions import ActionTransformer
from lib.policy import InverseActionPolicy
from lib.torch_util import default_device_type, set_default_torch_device
from agent import resize_image, AGENT_RESOLUTION


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

class IDMAgent:
    """
    Sugarcoating on the inverse dynamics model (IDM) used to predict actions Minecraft players take in videos.

    Functionally same as MineRLAgent.
    """
    def __init__(self, idm_net_kwargs, pi_head_kwargs, device=None):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = IDMActionMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        idm_policy_kwargs = dict(idm_net_kwargs=idm_net_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = InverseActionPolicy(**idm_policy_kwargs).to(device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)

    def _video_obs_to_agent(self, video_frames):
        imgs = [resize_image(frame, AGENT_RESOLUTION) for frame in video_frames]
        # Add time and batch dim
        imgs = np.stack(imgs)[None]
        agent_input = {"img": th.from_numpy(imgs).to(self.device)}
        return agent_input

    def _agent_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = {
            "buttons": agent_action["buttons"].cpu().numpy(),
            "camera": agent_action["camera"].cpu().numpy()
        }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def predict_actions(self, video_frames):
        """
        Predict actions for a sequence of frames.

        `video_frames` should be of shape (N, H, W, C).
        Returns MineRL action dict, where each action head
        has shape (N, ...).

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._video_obs_to_agent(video_frames)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        dummy_first = th.zeros((video_frames.shape[0], 1)).to(self.device)
        predicted_actions, self.hidden_state, _ = self.policy.predict(
            agent_input, first=dummy_first, state_in=self.hidden_state,
            deterministic=True
        )
        predicted_minerl_action = self._agent_action_to_env(predicted_actions)
        return predicted_minerl_action
