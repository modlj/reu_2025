# John Modl, iEdge 2025 REU

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
        # obs from self.env.step is (H, W, C). Permute to (C, H, W) for autoencoder.
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) 

        with torch.no_grad():
            reconstruction_tensor = self.autoencoder(obs_tensor)
        
        # For SSIM, convert (C, H, W) to (H, W, C)
        obs_np_for_ssim = obs_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstruction_np_for_ssim = reconstruction_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        ssim_score = compute_ssim(obs_np_for_ssim, reconstruction_np_for_ssim, data_range=1.0)
        intrinsic_reward = (1.0 - ssim_score) * self.intrinsic_reward_scale
        reward = intrinsic_reward # Override reward with intrinsic reward

        info['intrinsic_reward'] = intrinsic_reward 
        info['ssim_score'] = ssim_score
        return obs, reward, terminated, truncated, info