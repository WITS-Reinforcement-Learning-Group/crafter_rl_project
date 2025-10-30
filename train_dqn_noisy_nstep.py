# train_dqn_noisy_nstep.py
# Rainbow-style NoisyNet DQN + your NStep replay + existing env wrappers
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from env_make import make_vec_env_with_stack
from per_nstep_replay import NStepReplayBuffer
from noisy_cnn_policy import NoisyCnnPolicy

# --------------------------- CONFIG ---------------------------
CONFIG = {
    "env_id": "CrafterPartial-v1",
    "obs_size": 84,
    "frame_stack": 4,
    "n_envs": 1,                  # set 4–8 later for diversity
    "use_subproc": False,
    "start_method": "spawn",

    "total_timesteps": 1_000_000,
    "logdir": "logs/dqn_noisy_nstep_csv",
    "save_path": "dqn_noisy_nstep.zip",

    # N-step specifics
    "n_step": 5,
    "base_gamma": 0.99,

    # DQN hyperparams (stability for n=5)
    "buffer_size": 100_000,       # bump to 200k if RAM allows (with optimize_memory_usage=True)
    "learning_starts": 90_000,
    "batch_size": 64,
    "lr": 1e-4,
    "train_freq": 8,
    "target_update_interval": 6_000,

    # No epsilon-greedy when using NoisyNets
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01,

    "progress_bar": True,
    "csv_only": True,

    # Optional curiosity (OFF by default; train-only)
    "use_rnd": True,
    "rnd_eta": 0.02,
}
# --------------------------------------------------------------

# Housekeeping
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


class ResetNoisyCallback(BaseCallback):
    """Resample NoisyNet params periodically (and after each rollout)."""
    def __init__(self, every_steps: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.every_steps = every_steps
        self._last = 0

    def _on_rollout_end(self) -> None:
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "reset_noise"):
            self.model.policy.reset_noise()

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last) >= self.every_steps:
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "reset_noise"):
                self.model.policy.reset_noise()
            self._last = self.num_timesteps
        return True


def main():
    cfg = CONFIG

    # Optional: turn on RND exploration bonus for training via env vars
    if cfg["use_rnd"]:
        os.environ["USE_RND"] = "1"
        os.environ["RND_ETA"] = str(cfg["rnd_eta"])
    else:
        os.environ.pop("USE_RND", None)
        os.environ.pop("RND_ETA", None)

    # Build env with your standard wrappers
    env = make_vec_env_with_stack(
        env_id=cfg["env_id"],
        n_envs=cfg["n_envs"],
        size=cfg["obs_size"],
        frame_stack=cfg["frame_stack"],
        use_subproc=cfg["use_subproc"],
        start_method=cfg["start_method"],
    )

    # Align algorithm gamma with n-step returns rolled inside replay
    alg_gamma = cfg["base_gamma"] ** cfg["n_step"]

    model = DQN(
        NoisyCnnPolicy,                   # <-- noisy dueling policy
        env,
        buffer_size=cfg["buffer_size"],
        #optimize_memory_usage=True,       # remove if your SB3 build objects
        batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        train_freq=cfg["train_freq"],
        target_update_interval=cfg["target_update_interval"],
        gamma=alg_gamma,
        exploration_fraction=cfg["exploration_fraction"],  # 0.0
        exploration_final_eps=cfg["exploration_final_eps"],# 0.0
        learning_starts=cfg["learning_starts"],
        verbose=1,
        replay_buffer_class=NStepReplayBuffer,
        replay_buffer_kwargs=dict(n_step=cfg["n_step"], gamma=cfg["base_gamma"]),
    )

    logger_formats = ["csv"] if cfg["csv_only"] else ["stdout", "csv"]
    logger = configure(cfg["logdir"], logger_formats)
    model.set_logger(logger)

    cb = ResetNoisyCallback(every_steps=10000000)

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        progress_bar=cfg["progress_bar"],
        callback=cb,
    )

    model.save(cfg["save_path"])
    env.close()
    print(f"[NoisyNets + DQN + {cfg['n_step']}-step, γ_alg={alg_gamma:.6f}] Saved to: {cfg['save_path']}")


if __name__ == "__main__":
    main()
