# eval_crafter_metrics.py
import argparse, numpy as np
from shimmy import GymV21CompatibilityV0
import crafter
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack


def load_model(path):
    for cls in (PPO, DQN):
        try:
            return cls.load(path)
        except Exception:
            continue
    raise RuntimeError("Could not load model.")


# eval_crafter_metrics.py (only the env part shown)
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from shimmy import GymV21CompatibilityV0
import crafter

def make_eval_env(logdir, size=84, n_stack=4):
    # 1) Crafter (old Gym) + Recorder
    old = crafter.Env(reward=True)
    old = crafter.Recorder(old, logdir, save_stats=True, save_video=False, save_episode=True)

    # 2) Shimmy to Gymnasium API
    env = GymV21CompatibilityV0(env=old)

    # 3) Preprocess: grayscale + resize like training
    env = WarpFrame(env, width=size, height=size)

    # 4) Vectorize + FRAME STACK to match training (4 frames)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)  # <-- add this

    # 5) Channel order to CHW for SB3 policies
    vec_env = VecTransposeImage(vec_env)
    return vec_env



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--logdir", default="eval_logs/run1")
    args = p.parse_args()

    env = make_eval_env(args.logdir)
    model = load_model(args.model)

    ep_rewards, ep_lengths = [], []
    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        R, L = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            R += float(rewards)
            L += 1
            done = dones[0]
        ep_rewards.append(R)
        ep_lengths.append(L)

    print(f"Evaluation complete on {args.episodes} episodes")
    print(f"Avg reward: {np.mean(ep_rewards):.2f}")
    print(f"Avg steps: {np.mean(ep_lengths):.1f}")
    print(f"Logs (including Crafter achievements) → {args.logdir}")
    env.close()
