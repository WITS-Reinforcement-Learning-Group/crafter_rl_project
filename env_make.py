# env_make.py
from typing import Callable
import os
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from shimmy import GymV21CompatibilityV0

# Keep threads in check for env workers (harmless if single-env)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# Add this wrapper (place near other wrappers)
class _SeedCompat(gym.Wrapper):
    """Adds a .seed(seed) method for old-Gym envs so Gymnasium shim doesn't crash."""
    def seed(self, seed=None):
        # Prefer native seed if present
        if hasattr(self.env, "seed") and callable(getattr(self.env, "seed")):
            return self.env.seed(seed)
        # Otherwise, try Gymnasium-style reset(sep=seed) once
        try:
            self.env.reset(seed=seed)
        except TypeError:
            pass
        return [seed]

def _build_crafter_old_api(env_id: str):
    """
    Create the old-Gym Crafter env directly, then wrap it for Gymnasium.
    """
    import crafter
    if env_id == "CrafterPartial-v1":
        return crafter.Env(reward=True)
    elif env_id == "CrafterNoReward-v1":
        return crafter.Env(reward=False)
    else:
        raise ValueError(
            f"Unknown env_id '{env_id}'. Use 'CrafterPartial-v1' or 'CrafterNoReward-v1'."
        )


def _wrap_grayscale_resize(env: gym.Env, size: int) -> gym.Env:
    """
    Apply WarpFrame with a signature that matches the installed SB3 version.
    Prefers: WarpFrame(env, width, height, grayscale=True, keep_dim=True)
    Fallbacks: without grayscale/keep_dim as needed.
    """
    # Try newest signature first
    try:
        return WarpFrame(env, width=size, height=size, grayscale=True, keep_dim=True)
    except TypeError:
        pass
    # Try without keep_dim
    try:
        return WarpFrame(env, width=size, height=size, grayscale=True)
    except TypeError:
        pass
    # Try without grayscale (older SB3 defaults to grayscale=True)
    try:
        return WarpFrame(env, width=size, height=size)
    except TypeError:
        # As a last resort, import Gymnasium wrappers manually
        from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (size, size))
        return env


def make_crafter_env(
    env_id: str = "CrafterPartial-v1",
    size: int = 84,
) -> Callable[[], gym.Env]:
    """
    Factory for a single Crafter env (consistent with your baseline):
      old Gym crafter.Env -> GymV21CompatibilityV0 -> RecordEpisodeStatistics -> WarpFrame (grayscale+resize)
    """
    def _init():
        old_env = _build_crafter_old_api(env_id)
        env = GymV21CompatibilityV0(env=old_env)

        # Stats wrapper (handle older/newer signatures)
        try:
            env = RecordEpisodeStatistics(env, deque_size=1000)
        except TypeError:
            env = RecordEpisodeStatistics(env)

        # Grayscale + resize to (size, size), keep_dim if supported
        env = _wrap_grayscale_resize(env, size)
        return env

    return _init


def make_vec_env_with_stack(
    env_id,
    n_envs: int = 1,
    size: int = 84,
    frame_stack: int = 4,
    use_subproc: bool = False,
    start_method: str | None = None,
):
    """
    Create a vectorized env with consistent wrappers for string ids and callables:
        crafter.Env -> GymV21CompatibilityV0 -> RecordEpisodeStatistics -> WarpFrame
      then: VecTransposeImage (HWC->CHW) -> VecFrameStack(channels_first)
    """
    if callable(env_id):
        env_fns = [env_id for _ in range(n_envs)]
    else:
        factory = make_crafter_env(env_id=env_id, size=size)
        env_fns = [factory for _ in range(n_envs)]

    if use_subproc and n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec = DummyVecEnv(env_fns)

    # Keep order: transpose first, then stack (channels_first)
    vec = VecTransposeImage(vec)
    vec = VecFrameStack(vec, frame_stack, channels_order="first")
    return vec
