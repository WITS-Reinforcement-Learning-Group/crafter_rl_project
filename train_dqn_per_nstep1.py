# python train_dqn_per_nstep1.py --env_id CrafterPartial-v1 --total_timesteps 1000000
# Edit the CONFIG dict below; no need to pass CLI args.

import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from env_make import make_vec_env_with_stack
from per_nstep_replay import NStepReplayBuffer

# --------------------------- EDIT ME ---------------------------
CONFIG = {
    # Env/runtime
    "env_id": "CrafterPartial-v1",
    "obs_size": 84,
    "frame_stack": 4,
    "n_envs": 1,                 # For speed later, set to 4 or 8 (ensure NStep buffer is multi-env safe)
    "use_subproc": False,        # True if n_envs > 1 and you want parallel envs
    "start_method": None,        # "spawn" on Windows, "fork" on Linux (optional)

    "total_timesteps": 1_000_000,
    "logdir": "logs/dqn_nstep_csv1",
    "save_path": "dqn_nstep1.zip",

    # N-step specifics
    "n_step": 5,                 # <- recommended sweet spot on Crafter
    "base_gamma": 0.99,          # per-step gamma used to roll n-step rewards

    # DQN hyperparams
    "buffer_size": 100000,      # larger for n>=5
    "learning_starts":90000,   # warmup for n-step transitions
    "batch_size": 64,            # larger batch for more stable targets
    "lr": 7e-5,                  # slightly lower LR as n increases
    "train_freq": 8,             # fewer updates per env step smooths learning
    "target_update_interval": 8_000,  # sync target a bit faster for n>=5
    "exploration_fraction": 0.45,     # longer exploration
    "exploration_final_eps": 0.05,

    # Logging
    "progress_bar": True,        # set False for a tiny speed boost
    "csv_only": True             # only CSV logger (no stdout logger)
}
# ---------------------------------------------------------------

# Housekeeping for speed/stability (safe defaults)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


def main():
    cfg = CONFIG

    # Build env with identical wrappers to baseline
    env = make_vec_env_with_stack(
        env_id=cfg["env_id"],
        n_envs=cfg["n_envs"],
        size=cfg["obs_size"],
        frame_stack=cfg["frame_stack"],
        use_subproc=cfg["use_subproc"],
        start_method=cfg["start_method"],
    )

    # IMPORTANT: algorithm gamma must equal (base_gamma ** n_step) because
    # the N-step buffer already rolls rewards with base_gamma.
    alg_gamma = cfg["base_gamma"] ** cfg["n_step"]

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        train_freq=cfg["train_freq"],
        target_update_interval=cfg["target_update_interval"],
        gamma=alg_gamma,
        exploration_fraction=cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        learning_starts=cfg["learning_starts"],
        verbose=1,
        replay_buffer_class=NStepReplayBuffer,
        # NStep buffer needs base_gamma to roll R^(n); it stores (R^n, s_{t+n}, done_{t+n})
        replay_buffer_kwargs=dict(n_step=cfg["n_step"], gamma=cfg["base_gamma"]),
    )

    logger_formats = ["csv"] if cfg["csv_only"] else ["stdout", "csv"]
    logger = configure(cfg["logdir"], logger_formats)
    model.set_logger(logger)

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        progress_bar=cfg["progress_bar"],
    )

    model.save(cfg["save_path"])
    env.close()
    print(f"[DQN + {cfg['n_step']}-step, γ_alg={alg_gamma:.6f}] Saved to: {cfg['save_path']}")


if __name__ == "__main__":
    main()
