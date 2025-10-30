# train_dqn.py
import argparse
from stable_baselines3 import DQN
from env_make import make_vec_env_with_stack
from env_make import make_crafter_env
from stable_baselines3.common.logger import configure

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="CrafterPartial-v1")
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--total_timesteps", type=int, default=5_000_000)
    p.add_argument("--save_path", type=str, default="dqn_crafter_baseline")
    args = p.parse_args()

    env = make_vec_env_with_stack(args.env_id, n_envs=1, size=args.size, frame_stack=args.stack)

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

    csv_dir = "logs/dqn_csv"  # or make it a flag
    new_logger = configure(csv_dir, ["csv"])  # writes logs/progress.csv
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.save_path)
    env.close()
    print(f"[DQN] Saved to: {args.save_path}")

if __name__ == "__main__":
    main()
