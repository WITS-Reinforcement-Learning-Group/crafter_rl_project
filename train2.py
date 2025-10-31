#Improvement 1

import argparse
import torch
import torch.nn as nn
import gymnasium as gym
import crafter
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn


# --- Custom CNN Feature Extractor with Image Normalization ---
class ImprovedCNNExtractor(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor for Crafter (64x64x3 visuals).
    Incorporates:
      - Larger first conv for wide receptive field
      - BatchNorm for stability
      - Slightly deeper conv stack
      - 512-dim latent space for richer features
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=512)

        n_input_channels = observation_space.shape[0]  # should be (C,H,W) after VecTransposeImage

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten()
        )

        # --- Compute output size dynamically ---
        with torch.no_grad():
            dummy = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_out = self.cnn(dummy)
            cnn_out_dim = cnn_out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1)  # small regularization to prevent overfitting
        )

        # --- Weight initialization ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations / 255.0  # normalize pixel values
        return self.linear(self.cnn(observations))


# --- Main training ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='logdir/crafter_ppo_improved')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    lr_schedule = get_linear_fn(0.0003, 0.00003, args.steps)  # (start, end, total_timesteps)

    # --- Crafter environment ---
    env = crafter.Env(reward=True)
    env = crafter.Recorder(
        env, args.outdir,
        save_stats=True,
        save_video=False,
        save_episode=False,
    )


    env = GymV21CompatibilityV0(env=env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)  # ensure (C,H,W) order

    # --- PPO model with custom CNN ---
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=lr_schedule,
        verbose=1,
        tensorboard_log=f"{args.outdir}/tensorboard",
        policy_kwargs=dict(
            features_extractor_class=ImprovedCNNExtractor
        ),
    )

    # --- Train ---
    model.learn(total_timesteps=args.steps)

    # --- Save model ---
    model.save(f"{args.outdir}/ppo_crafter_improved")


if __name__ == "__main__":
    main()
