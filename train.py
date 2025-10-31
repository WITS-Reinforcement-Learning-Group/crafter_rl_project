#Baseline PPO 

import argparse
import crafter
from stable_baselines3 import PPO
from shimmy import GymV21CompatibilityV0
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_ppo')
parser.add_argument('--steps', type=int, default=10000)  # small local test
args = parser.parse_args()

# --- Create Crafter environment directly ---
env = crafter.Env(reward=True)  # no Gym registration
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)

# --- Make it compatible with Gymnasium ---
env = GymV21CompatibilityV0(env=env)

# --- PPO model ---
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=f"{args.outdir}/tensorboard"
)

# --- Train briefly ---
model.learn(total_timesteps=args.steps)

# --- Save ---
model.save(f"{args.outdir}/ppo_crafter_test")

# --- Quick rollout ---
eval_env = Monitor(env)
print("🔍 Evaluating trained policy over 10 episodes...")
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

print(f"✅ Evaluation complete! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

