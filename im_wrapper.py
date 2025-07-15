import gymnasium as gym
import torch
from autoencoder import Autoencoder, compute_ssim
import numpy as np

class AutoencoderWrapper(gym.Wrapper):
    def __init__(self, env, autoencoder: Autoencoder, device, intrinsic_reward_scale=1.0):
        super().__init__(env)
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.device = device
        self.intrinsic_reward_scale = intrinsic_reward_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            reconstruction_tensor = self.autoencoder(obs_tensor)
        obs_np = obs_tensor.squeeze(0).cpu().numpy()

