import gym as old_gym
import crafter
import torch
import torch.nn as nn
import numpy as np
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import argparse

try:
    old_gym.register(
        id='CrafterPartial-v1',
        entry_point=crafter.Env,
        disable_env_checker=True
    )
except old_gym.error.Error:
    pass

numDict = {}
numDict[0] = np.load('nums/0.npy')
numDict[1] = np.load('nums/1.npy')
numDict[2] = np.load('nums/2.npy')
numDict[3] = np.load('nums/3.npy')
numDict[4] = np.load('nums/4.npy')
numDict[5] = np.load('nums/5.npy')
numDict[6] = np.load('nums/6.npy')
numDict[7] = np.load('nums/7.npy')
numDict[8] = np.load('nums/8.npy')
numDict[9] = np.load('nums/9.npy')


def extractResourceBar(obs):
    alteredObs = obs.copy()
    blockSize = 7
    health = alteredObs[7*blockSize:7*blockSize+7,0*blockSize:0*blockSize+7].copy()
    food = alteredObs[7*blockSize:7*blockSize+7,1*blockSize:1*blockSize+7].copy()
    water= alteredObs[7*blockSize:7*blockSize+7,2*blockSize:2*blockSize+7].copy()
    stamina = alteredObs[7*blockSize:7*blockSize+7,3*blockSize:3*blockSize+7].copy()
    return health,food,water,stamina

def extractNum(resource):
    alteredResource = resource.copy()
    height, width, _ = resource.shape
    for i in range(height):
        for j in range(width):
            if np.array_equal(resource[i, j], [255, 255, 255]) and i >= 2 and j >= 2:
                alteredResource[i, j] = [255, 255, 255]
            else:
                alteredResource[i, j] = [0, 0, 0]
    return alteredResource

def extractResourceNum(resource):
    for i in range(10):
        if np.array_equal(resource, numDict[i]):
            return i
    return 0

def extractResourceVec(obs):
    health, food, water, stamina = extractResourceBar(obs)
    return [
        extractResourceNum(extractNum(health)),
        extractResourceNum(extractNum(food)),
        extractResourceNum(extractNum(water)),
        extractResourceNum(extractNum(stamina))
    ]

def preprocess_obs(obs):
    obs_np = np.array(obs, dtype=np.float32) / 255.0
    obs_np = np.transpose(obs_np, (2, 0, 1))
    obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
    return obs_tensor


class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, num_resources=4):
        super(PolicyNetwork, self).__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        combined_input_size = (32 * 6 * 6) + num_resources
        self.final_layers = nn.Sequential(
            nn.Linear(combined_input_size, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, image_input, resource_input):
        cnn_output = self.cnn_branch(image_input)
        combined = torch.cat((cnn_output, resource_input), dim=1)
        return self.final_layers(combined)


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='logdir/crafter_ga_imp3/final/best_ga_policy.pth')
parser.add_argument('--outdir', default='logdir/crafter_ga_eval/imp3/final')
args = parser.parse_args()

print(f"--- Loading model from: {args.model_path} ---")
model = PolicyNetwork(num_actions=17)
model.load_state_dict(torch.load(args.model_path))
model.eval()

print(f"--- Setting up environment for demo ---")
print(f"--- Saving video/episode to: {args.outdir} ---")

env = old_gym.make('CrafterPartial-v1', seed=13)
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
    resource_vec = extractResourceVec(obs)
    resource_tensor = torch.tensor(resource_vec, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action_logits = model(obs_tensor, resource_tensor)
    
    action = torch.argmax(action_logits, dim=1).item()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

print("--- Episode Finished ---")
print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward:.2f}")
print(f"Episode data saved in '{args.outdir}' directory.")

env.close()
