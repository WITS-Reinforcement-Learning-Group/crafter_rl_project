# noisy_cnn_policy.py
# Rainbow-style: NatureCNN backbone + dueling NoisyNet head for SB3 DQN
import math
import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import NatureCNN


# --------- Factorized Noisy Linear (Fortunato et al., 2018) ----------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.register_buffer("weight_eps", th.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(th.empty(out_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))
        self.register_buffer("bias_eps", th.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int):
        x = th.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(th.ger(eps_out, eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            # resample factorized noise every forward pass (both acting & learning)
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return th.nn.functional.linear(x, weight, bias)


class NoisyDuelingHead(nn.Module):
    """Dueling head built with NoisyLinear layers."""
    def __init__(self, feature_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.adv_fc1 = NoisyLinear(feature_dim, hidden)
        self.adv_fc2 = NoisyLinear(hidden, n_actions)
        self.val_fc1 = NoisyLinear(feature_dim, hidden)
        self.val_fc2 = NoisyLinear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, features):
        adv = self.act(self.adv_fc1(features))
        adv = self.adv_fc2(adv)
        val = self.act(self.val_fc1(features))
        val = self.val_fc2(val)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class NoisyQNetwork(nn.Module):
    """
    Full Q-network: NatureCNN backbone + dueling Noisy head.
    Conforms to SB3's QNetwork expectations (forward, _predict, set_training_mode).
    """
    def __init__(self, observation_space, action_space, features_dim: int = 512):
        super().__init__()
        self.features_extractor = NatureCNN(observation_space, features_dim=features_dim)
        self.q_head = NoisyDuelingHead(features_dim, action_space.n)

    def forward(self, obs):
        # Ensure correct dtype/range for CNN
        # SB3 normally normalizes to float32 in [0,1]; do it here since we own the backbone.
        if obs.dtype == th.uint8:
            obs = obs.float().div_(255.0)
        elif obs.dtype != th.float32:
            obs = obs.float()

        features = self.features_extractor(obs)
        return self.q_head(features)

    # --- SB3 compatibility hooks ---
    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
        self.features_extractor.train(mode)

    @th.no_grad()
    def _predict(self, obs, deterministic: bool = True):
        if obs.dtype == th.uint8:
            obs = obs.float().div_(255.0)
        elif obs.dtype != th.float32:
            obs = obs.float()
        q_values = self.forward(obs)
        actions = q_values.argmax(dim=1).to(th.long)
        return actions

    def reset_noise(self):
        self.q_head.reset_noise()



class NoisyCnnPolicy(DQNPolicy):
    """
    SB3 DQNPolicy that uses NoisyQNetwork for q_net/q_net_target.
    Exposes reset_noise() for callbacks.
    """
    def make_q_net(self) -> nn.Module:
        return NoisyQNetwork(self.observation_space, self.action_space, features_dim=512)

    def reset_noise(self):
        if hasattr(self, "q_net"):
            self.q_net.reset_noise()
        if hasattr(self, "q_net_target"):
            self.q_net_target.reset_noise()
