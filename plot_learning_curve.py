import argparse, pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/dqn_csv/progress.csv")
    ap.add_argument("--out", default="dqn_learning_curve.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # common SB3 columns:
    # 'time/total_timesteps', 'rollout/ep_rew_mean', 'rollout/ep_len_mean'
    x = df["time/total_timesteps"]
    fig = plt.figure(figsize=(7,4.5))

    if "rollout/ep_rew_mean" in df:
        plt.plot(x, df["rollout/ep_rew_mean"], label="Episode reward")
    if "rollout/ep_len_mean" in df:
        plt.plot(x, df["rollout/ep_len_mean"], label="Episode length")

    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.title("Learning Curve (DQN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
