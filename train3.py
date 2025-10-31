#Improvement 2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import crafter
import numpy as np
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
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


# --- Intrinsic Curiosity Module (ICM) ---
class CuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for exploration.
    
    Consists of:
    1. Feature encoder: Maps observations to feature space
    2. Inverse model: Predicts action from (state, next_state) features
    3. Forward model: Predicts next_state features from (state, action)
    
    Intrinsic reward = prediction error of forward model
    """
    
    def __init__(self, observation_space, action_space, feature_dim=256, device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_space.n
        
        # --- Feature Encoder (shared) ---
        n_input_channels = observation_space.shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute encoder output size
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape).to(device)
            encoder_out_dim = self.encoder(dummy).shape[1]
        
        self.feature_projection = nn.Sequential(
            nn.Linear(encoder_out_dim, feature_dim),
            nn.ReLU()
        )
        
        # --- Inverse Model: predicts action from (phi(s), phi(s')) ---
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        # --- Forward Model: predicts phi(s') from (phi(s), a) ---
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        self.to(device)
    
    def encode(self, obs):
        """Encode observation to feature space"""
        obs = obs.float() / 255.0  # normalize
        return self.feature_projection(self.encoder(obs))
    
    def forward(self, obs, next_obs, actions):
        """
        Compute intrinsic reward and losses
        
        Returns:
            intrinsic_reward: curiosity bonus (forward model prediction error)
            inverse_loss: loss for inverse model
            forward_loss: loss for forward model
        """
        # Encode observations
        phi_s = self.encode(obs)
        phi_s_next = self.encode(next_obs)
        
        # Inverse model: predict action
        pred_actions = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=1))
        inverse_loss = F.cross_entropy(pred_actions, actions)
        
        # Forward model: predict next state features
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        pred_phi_s_next = self.forward_model(torch.cat([phi_s, actions_onehot], dim=1))
        
        # Intrinsic reward = prediction error (normalized)
        forward_loss = F.mse_loss(pred_phi_s_next, phi_s_next.detach(), reduction='none').mean(dim=1)
        intrinsic_reward = forward_loss.detach()
        
        return intrinsic_reward, inverse_loss, forward_loss.mean()


# --- Curiosity Callback for Training ---
class CuriosityCallback(BaseCallback):
    """
    Callback to:
    1. Compute intrinsic rewards using curiosity module
    2. Add them to extrinsic rewards
    3. Train the curiosity module
    """
    
    def __init__(self, curiosity_module, intrinsic_weight=0.01, verbose=0):
        super().__init__(verbose)
        self.curiosity_module = curiosity_module
        self.intrinsic_weight = intrinsic_weight
        self.curiosity_optimizer = torch.optim.Adam(curiosity_module.parameters(), lr=1e-3)
        
        # Tracking
        self.total_intrinsic_reward = 0
        self.total_extrinsic_reward = 0
        self.curiosity_updates = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        Compute intrinsic rewards and train curiosity module.
        """
        # Get rollout buffer
        rollout_buffer = self.model.rollout_buffer
        
        # Convert to tensors and handle shape
        obs = torch.FloatTensor(rollout_buffer.observations).to(self.curiosity_module.device)
        # Remove extra dimension if present: [batch, 1, C, H, W] -> [batch, C, H, W]
        if obs.dim() == 5 and obs.shape[1] == 1:
            obs = obs.squeeze(1)
        actions = torch.LongTensor(rollout_buffer.actions.squeeze()).to(self.curiosity_module.device)
        
        # Get next observations (shift by 1)
        next_obs = torch.cat([obs[1:], obs[-1:]], dim=0)
        
        # Compute intrinsic rewards and losses
        with torch.no_grad():
            intrinsic_rewards, _, _ = self.curiosity_module(obs, next_obs, actions)
        
        # Normalize intrinsic rewards (optional but recommended)
        intrinsic_rewards = (intrinsic_rewards - intrinsic_rewards.mean()) / (intrinsic_rewards.std() + 1e-8)
        
        # Add intrinsic rewards to buffer
        intrinsic_rewards_np = intrinsic_rewards.cpu().numpy()
        rollout_buffer.rewards += self.intrinsic_weight * intrinsic_rewards_np.reshape(-1, 1)
        
        # Train curiosity module
        self.curiosity_optimizer.zero_grad()
        _, inverse_loss, forward_loss = self.curiosity_module(obs, next_obs, actions)
        curiosity_loss = inverse_loss + forward_loss
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        
        # Logging
        self.total_intrinsic_reward += intrinsic_rewards_np.sum()
        self.total_extrinsic_reward += rollout_buffer.rewards.sum()
        self.curiosity_updates += 1
        
        if self.verbose > 0 and self.curiosity_updates % 10 == 0:
            print(f"[Curiosity] Updates: {self.curiosity_updates}, "
                  f"Avg Intrinsic: {self.total_intrinsic_reward / self.curiosity_updates:.4f}, "
                  f"Inverse Loss: {inverse_loss.item():.4f}, "
                  f"Forward Loss: {forward_loss.item():.4f}")


# --- Main training ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='logdir/crafter_ppo_curiosity')
    parser.add_argument('--steps', type=int, default=3000000)
    parser.add_argument('--curiosity_weight', type=float, default=0.01, 
                        help='Weight of intrinsic reward (default: 0.01)')
    parser.add_argument('--no_curiosity', action='store_true',
                        help='Disable curiosity module')
    args = parser.parse_args()

    # --- Device detection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

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
        policy_kwargs=dict(
            features_extractor_class=ImprovedCNNExtractor
        ),
        device=device,  # ✅ Use detected device
    )

    # --- Setup curiosity module ---
    callbacks = []
    if not args.no_curiosity:
        print(f"🔍 Enabling Curiosity Module (weight={args.curiosity_weight})")
        curiosity_module = CuriosityModule(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=256,
            device=device  # ✅ Use detected device
        )
        curiosity_callback = CuriosityCallback(
            curiosity_module=curiosity_module,
            intrinsic_weight=args.curiosity_weight,
            verbose=1
        )
        callbacks.append(curiosity_callback)
    else:
        print("⚠️  Curiosity module disabled")

    # --- Train ---
    model.learn(total_timesteps=args.steps, callback=callbacks)

    # --- Save model ---
    model.save(f"{args.outdir}/ppo_crafter_curiosity")
    print(f"✅ Model saved to {args.outdir}/ppo_crafter_curiosity")


if __name__ == "__main__":
    main()