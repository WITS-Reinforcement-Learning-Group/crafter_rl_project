# crafter_gym_wrapper.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import crafter  # pip install crafter
except Exception as e:
    raise ImportError("Please install the 'crafter' package: pip install crafter") from e


class CrafterGymnasium(gym.Env):
    """Wrap crafter.Env to Gymnasium API (terminated/truncated)."""
    metadata = {"render_modes": []}

    def __init__(self, reward=True, seed=None):
        super().__init__()
        self._reward_flag = reward
        self._env = crafter.Env(reward=reward, seed=seed)
        h, w, c = self._env.observation_space.shape
        self.observation_space = spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
        self.action_space = spaces.Discrete(self._env.action_space.n)

    def reset(self, *, seed=None, options=None):
        # crafter.Env has no .seed(); re-create if Gymnasium passes a seed
        if seed is not None:
            self._env = crafter.Env(reward=self._reward_flag, seed=seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = False
        return obs, float(rew), terminated, truncated, info

    def close(self):
        self._env.close()


def ensure_crafter_registered():
    from gymnasium.envs.registration import register, registry
    if "CrafterPartial-v1" not in registry:
        register(
            id="CrafterPartial-v1",
            entry_point=lambda **kwargs: CrafterGymnasium(reward=True, **kwargs),
        )
