# train_dqn.py
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from env_make import make_vec_env_with_stack, make_crafter_env

# --- Reward shaping wrapper: small per-step survival bonus ---
from gymnasium import RewardWrapper
class SurvivalBonus(RewardWrapper):
    """
    Adds a small per-step reward to encourage survival/exploration.
    Works with Gymnasium's (obs, reward, terminated, truncated, info) API.
    """
    def __init__(self, env, bonus: float = 0.01):
        super().__init__(env)
        self.bonus = float(bonus)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        r += self.bonus
        # (Optional) track cumulative shaping added this episode
        info["survival_bonus"] = info.get("survival_bonus", 0.0) + self.bonus
        return obs, r, terminated, truncated, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="CrafterPartial-v1")
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--save_path", type=str, default="dqn_crafter_rShape")
    p.add_argument("--survival_bonus", type=float, default=0.01)  # <--- bonus as a flag
    args = p.parse_args()

    # Build a factory that creates a SINGLE wrapped env (reward shaping happens before vec)
    def make_wrapped_env():
        base = make_crafter_env(args.env_id, size=args.size)

        # If make_crafter_env returns a factory, instantiate it:
        env = base() if callable(base) else base

        # Now env is a real gymnasium.Env
        env = SurvivalBonus(env, bonus=args.survival_bonus)
        return env



    # Vectorize + frame-stack using your helper (it should accept a callable/env-factory)
    env = make_vec_env_with_stack(make_wrapped_env, n_envs=1, size=args.size, frame_stack=args.stack)

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=100_000,
        learning_rate=1e-4,
        batch_size=32,
        train_freq=4,
        target_update_interval=10_000,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
    )

    csv_dir = "logs/dqn_rShape_csv"  # or make it a flag
    new_logger = configure(csv_dir, ["csv"])  # writes logs/progress.csv
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.save_path)
    env.close()
    print(f"[DQN] Saved to: {args.save_path}")

if __name__ == "__main__":
    main()
