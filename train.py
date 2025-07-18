import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid_env import GridEnv
from minigrid.wrappers import ImgObsWrapper # For pixel observations
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from autoencoder import Autoencoder
from im_wrapper import AutoencoderWrapper

# Custom CNN necessary for DQN integration in MiniGrid
class GridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = torch.nn.Sequential(
            # First convolutional layer (9 x 9 with padding = 1)
            torch.nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # Second convolutional layer (9 x 9 with padding = 1)
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float().permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
            n_flatten = self.cnn(dummy_input).shape[1]
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim), 
            torch.nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert observations from (B, H, W, C) to (B, C, H, W)
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))
    
class AutoencoderTrainCallback(BaseCallback):
    def __init__(self, autoencoder: Autoencoder, optimizer, loss_fun, device,
                 buffer_size: int = 10000, train_freq: int = 100, batch_size: int = 64, verbose: int = 0):
        super().__init__(verbose)
        self.autoencoder = autoencoder
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.total_steps = 0
         
    def _on_step(self) -> bool:
        # Collect current observation
        current_obs = self.training_env.get_ordered_obs()[0]
        self.buffer.append(current_obs)
        self.total_steps += 1
        # Periodically train CAE
        if self.total_steps % self.train_freq == 0 and len(self.buffer) >= self.batch_size:
            self._train_autoencoder()
        return True
    
    def _train_autoencoder(self):
        # Sample batch from buffer
        indices = np.random.choice(len(self.buffer), self.batch_size, replace = False)
        batch_obs = [self.buffer[i] for i in indices]
        # Convert numpy arrays to tensor
        # (H, W, C) -> (B, C, H, W)
        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        self.optimizer.zero_grad()
        reconstructed_obs = self.autoencoder(obs_tensor)
        loss = self.loss_fn(reconstructed_obs, obs_tensor)
        loss.backward()
        self.optimizer.step()
        if self.verbose > 0:
            print(f"Autoencoder trained at step {self.total_steps}, Loss: {loss.item():.4f}")

def main():
    print("--- Setting up & training agent ---")
    # Device config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    autoencoder_input_channels = 3 # RGB channels
    autoencoder = Autoencoder(input_channels=autoencoder_input_channels).to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4) # Lower LR for AE
    autoencoder_loss_fn = torch.nn.MSELoss() # Or nn.L1Loss(), might need to change

    # Establish environment
    env = GridEnv(render_mode="rgb_array")  
    env = ImgObsWrapper(env)  # Convert to pixel observations
    env = AutoencoderWrapper(env, autoencoder, device, intrinsic_reward_scale=1.0)

    env = DummyVecEnv([lambda: env]) 
    env = VecMonitor(env, "./logs/") # Log training progress

    # RL agent instantiation
    model = DQN(
        "CnnPolicy",  
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        device=device,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,  
            features_extractor_kwargs=dict(features_dim=256)
        )
    )

#####################################################################
    # Agent training
    print("Starting training...")
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps)  
    print("Training finished!")

    # Save the trained model
    model.save("dqn_minigrid_model")
    print("Model saved as 'dqn_minigrid_model.zip'")

    # Training eval
    print(" --- Evaluating trained agent ---")
    eval_env = GridEnv(render_mode="human")
    eval_env = ImgObsWrapper(eval_env)

    obs, info = eval_env.reset()

    # Basic flow
    for episode in range(5):
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render()
            total_reward += reward
            done = terminated or truncated
            if done:
                print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                obs, info = eval_env.reset()
    eval_env.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()



