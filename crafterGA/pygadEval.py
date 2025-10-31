import gym as old_gym
import crafter
import torch
import torch.nn as nn
import numpy as np
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import argparse

# --- Register the environment ONCE, at the top ---
# This must be the same as in your training script.
try:
    old_gym.register(
        id='CrafterPartial-v1',
        entry_point=crafter.Env,
        disable_env_checker=True
    )
except old_gym.error.Error:
    pass # Environment is already registered


# --- Step 1: Define the Policy Network ---
# We need to define the exact same network architecture as the one used
# during training so we can load the saved weights into it.
class PolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            # Input: (batch_size, 3, 64, 64)
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def preprocess_obs(obs):
    obs_np = np.array(obs, dtype=np.float32) / 255.0
    obs_np = np.transpose(obs_np, (2, 0, 1))
    obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
    return obs_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='logdir/crafter_ga_base/final/best_ga_policy.pth')
parser.add_argument('--outdir', default='logdir/crafter_ga_eval/base')
args = parser.parse_args()

print(f"--- Loading model from: {args.model_path} ---")

model = PolicyNetwork(num_actions=17)

model.load_state_dict(torch.load(args.model_path))

model.eval()

print(f"--- Setting up environment for demo ---")
print(f"--- Saving video to: {args.outdir} ---")

env = old_gym.make('CrafterPartial-v1', seed=42)
env = crafter.Recorder(
  env,
  args.outdir,
  save_stats=True,
  save_video=False,
  save_episode=True,
)
env = GymV21CompatibilityV0(env=env)

obs, info = env.reset()
done = False
total_reward = 0
step_count = 0

print("\n--- Running Demo Episode ---")
while not done:
    obs_tensor = preprocess_obs(obs)
    
    with torch.no_grad():
        action_logits = model(obs_tensor)
    
    action = torch.argmax(action_logits, dim=1).item()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

print("--- Episode Finished ---")
print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward:.2f}")
print(f"Video of the episode saved in '{args.outdir}' directory.")

env.close()
