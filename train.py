# John Modl, iEdge 2025 REU

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import minigrid_env
from minigrid_env import GridEnv
from minigrid.wrappers import ImgObsWrapper # For pixel observations
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from autoencoder import Autoencoder
from im_wrapper import AutoencoderWrapper
import argparse
import random
import os
from torch.utils.tensorboard import SummaryWriter

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
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Corrected line
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()).float().unsqueeze(0)
            n_flatten = self.cnn(dummy_input).shape[1]
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim), 
            torch.nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert observations from (B, H, W, C) to (B, C, H, W)
        # observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))
    
class AutoencoderTrainCallback(BaseCallback):
    def __init__(self, autoencoder: Autoencoder, optimizer, loss_fun, device,
                 buffer_size: int = 10000, train_freq: int = 100, batch_size: int = 64, verbose: int = 0):
        super().__init__(verbose)
        self.autoencoder = autoencoder
        self.optimizer = optimizer
        self.loss_fn = loss_fun 
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.total_steps = 0
         
    def _on_step(self) -> bool:
        # Collect current observation
        current_obs = self.locals['new_obs'][0] # Assuming single environment in DummyVecEnv
        self.buffer.append(current_obs)
        self.total_steps += 1
        # Periodically train CAE
        if self.total_steps % self.train_freq == 0 and len(self.buffer) >= self.batch_size:
            self._train_autoencoder()
        
        # Log exploration metrics
        # self._log_exploration_metrics()
        percentage_visited = self._log_exploration_metrics() # Have this function return the value

        # Check for cutoff condition
        if percentage_visited is not None and percentage_visited >= 100.0:
            print(f"Stopping training: 100% exploration reached at step {self.total_steps}.")
            return False # This will stop the model.learn() loop
        
        return True


    def _log_exploration_metrics(self) -> float:
        # Access the original environment to get visit_counts
        # self.training_env is VecEnv
        
        
        visit_counts = self.training_env.get_attr('visit_counts')[0]
        
        
        num_visited_cells = np.sum(visit_counts > 0)
        
        
        total_navigable_cells = self.training_env.get_attr('num_navigable_cells')[0]
        
        
        # total_navigable_cells = (self.training_env.get_attr('width')[0] - 2) * (self.training_env.get_attr('height')[0] - 2)
        
        if total_navigable_cells > 0:
            percentage_visited = (num_visited_cells / total_navigable_cells) * 100
        else:
            percentage_visited = 0.0

        if self.verbose > 0:
            print(f"Step {self.total_steps}: Visited {num_visited_cells}/{total_navigable_cells} cells ({percentage_visited:.2f}%)")
        
        # Log to TensorBoard
        if self.logger:
            self.logger.record("exploration/percentage_visited_cells", percentage_visited)
            self.logger.record("exploration/num_visited_cells", num_visited_cells)
            self.logger.record("exploration/total_navigable_cells", total_navigable_cells)
            
            if 'infos' in self.locals and self.locals['infos'] and 'ssim_score' in self.locals['infos'][0] and 'intrinsic_reward' in self.locals['infos'][0]:
                 self.logger.record("metrics/ssim_score", self.locals['infos'][0]['ssim_score'])
                 self.logger.record("metrics/intrinsic_reward", self.locals['infos'][0]['intrinsic_reward'])
        return percentage_visited

    














    def _train_autoencoder(self):
        # Sample batch from buffer
        indices = np.random.choice(len(self.buffer), self.batch_size, replace = False)
        batch_obs = [self.buffer[i] for i in indices]
        
        # Debugging prints:
        print(f"Shape of one sample in batch_obs (H, W, C): {batch_obs[0].shape}") 
        
        obs_np_array = np.array(batch_obs)
        print(f"Shape of np.array(batch_obs) (B, H, W, C): {obs_np_array.shape}")
        
        # Convert numpy arrays to tensor
        # (B, H, W, C) -> (B, C, H, W)
        obs_tensor = torch.as_tensor(obs_np_array, dtype=torch.float32, device=self.device)
        print(f"Shape of obs_tensor after permute (B, C, H, W): {obs_tensor.shape}")
        
        self.optimizer.zero_grad()
        reconstructed_obs = self.autoencoder(obs_tensor)
        loss = self.loss_fn(reconstructed_obs, obs_tensor)
        loss.backward()
        self.optimizer.step()
        if self.logger:
            self.logger.record("train/cae_loss", loss.item())
        if self.verbose > 0:
            print(f"Autoencoder trained at step {self.total_steps}, Loss: {loss.item():.4f}")

def get_args():
    parser = argparse.ArgumentParser(description="Train RL agent with different intrinsic motivation strategies.")
    parser.add_argument("--mode", type=str, default="autoencoder_curiosity",
                        choices=["autoencoder_curiosity", "random_exploration", "count_based_curiosity"],
                        help="Experiment mode: autoencoder_curiosity (default), random_exploration, or count_based_curiosity")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total number of environment steps to train for")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def random_exploration(total_timesteps = 1000000):
    print("Running Random Exploration Baseline...")
    env = gym.make('MiniGrid-Custom-Grid-v0', size=27)
    current_timesteps = 0
    visited_cells = set()
    initial_obs, info = env.reset()
    total_navigable_cells = env.unwrapped.num_navigable_cells
    print(f"Total navigable cells: {total_navigable_cells}")

    log_dir = "./logs/random_exploration/"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    obs = env.reset()
    episode_rewards = 0
    num_episodes = 0

    for step in range(total_timesteps):
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_rewards += reward

        # Calculate visited percentage using the correct total
        visit_counts = env.unwrapped.visit_counts
        num_visited_cells = np.sum(visit_counts > 0)
        
        # 2. Use the correct denominator in the calculation
        if total_navigable_cells > 0:
            percentage_visited = (num_visited_cells / total_navigable_cells) * 100
        else:
            percentage_visited = 0.0

        writer.add_scalar("exploration/percentage_visited_cells", percentage_visited, step)
        
        if done:
            obs = env.reset()
            writer.add_scalar("rollout/ep_rew_mean", episode_rewards, num_episodes)
            num_episodes += 1
            episode_rewards = 0
            # Optional: Add a cutoff for random exploration as well
            if percentage_visited >= 100:
                print(f"Random exploration reached 100% at step {step}. Stopping.")
                break

    writer.close()
    env.close()
    print("Finished random exploration.")




def main():
    args = get_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("--- Setting up & training agent ---")
    # Device config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    autoencoder = None
    autoencoder_optimizer = None
    autoencoder_loss_fn = None
    autoencoder_callback = None

    env_kwargs = {"render_mode": "rgb_array", "enable_intrinsic_reward": False}

    # if args.mode == "count_based_curiosity":
    #     env_kwargs["enable_intrinsic_reward"] = True

    env = GridEnv(**env_kwargs)
    env = ImgObsWrapper(env)

    if args.mode == "autoencoder_curiosity":
        autoencoder_input_channels = 3 # RGB channels
        
        # Get observation space shape to pass to Autoencoder
        # After ImgObsWrapper, observation space is (H, W, C)
        obs_height, obs_width, _ = env.observation_space.shape 
        
        autoencoder = Autoencoder(input_channels=autoencoder_input_channels,
                                  input_height=obs_height,
                                  input_width=obs_width).to(device)
        autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4) # Lower LR for AE
        autoencoder_loss_fn = torch.nn.MSELoss() # Or nn.L1Loss(), might need to change
    
        env = AutoencoderWrapper(env, autoencoder, device, intrinsic_reward_scale=1.0)

        ae_buffer_size = 10000
        ae_train_freq = 100
        ae_batch_size = 64

        autoencoder_callback = AutoencoderTrainCallback(
            autoencoder=autoencoder,
            optimizer=autoencoder_optimizer,
            loss_fun=autoencoder_loss_fn,
            device=device,
            buffer_size=ae_buffer_size,
            train_freq=ae_train_freq,
            batch_size=ae_batch_size,
            verbose=1
        )
    
    env = DummyVecEnv([lambda: env])
    log_dir = f"./logs/{args.mode}_{args.seed}/" # Create a unique log directory for each mode/seed
    env = VecMonitor(env, log_dir) # Log training progress

    if args.mode == "random_exploration":
        random_exploration(args.total_timesteps)
        # print("Running Random Exploration Baseline...")
        # # Simulate training loop for logging
        # obs = env.reset()[0]
        # for step in range(args.total_timesteps):
        #     action = [env.action_space.sample()] # Take a random action
        #     obs, reward, dones, info = env.step(action)
        #     done = dones[0]
        #     if step % 500 == 0: 
        #         # visit_counts obtained from original GridEnv
        #         visit_counts = env.get_attr('visit_counts')[0]
        #         num_visited_cells = np.sum(visit_counts > 0)
        #         total_navigable_cells = (env.get_attr('width')[0] - 2) *                                         (env.get_attr('height')[0] - 2)
        #         percentage_visited = (num_visited_cells / total_navigable_cells) * 100 if total_navigable_cells > 0 else 0.0
        #         print(f"Random Step {step}: Visited {num_visited_cells}/{total_navigable_cells} cells ({percentage_visited:.2f}%)")

        #     if done:
        #         obs = env.reset()[0]
        # print("Random Exploration finished!")

    else: # autoencoder_curiosity or count_based_curiosity
        model_name = f"dqn_minigrid_{args.mode}_seed{args.seed}"
        print(f"Running {args.mode} mode. Model will be saved as {model_name}.zip")
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
            exploration_fraction=0.05, # Can set lower for intrinsic, or adapt for specific needs
            exploration_final_eps=0.01, # Or remove if intrinsic reward is the ONLY driver
            verbose=1,
            device=device,
            policy_kwargs=dict(
                features_extractor_class=GridCNN,  
                features_extractor_kwargs=dict(features_dim=256)
            )
        )

        print("Starting training...")
        model.learn(total_timesteps=args.total_timesteps, callback=autoencoder_callback)
        print("Training finished!")
        model.save(model_name)
        print(f"Model saved as '{model_name}.zip'")

    # Evaluation
    print(" --- Basic Evaluation (headless mode) ---")
    eval_env = GridEnv(render_mode="rgb_array")
    eval_env = ImgObsWrapper(eval_env)
    
    # Evaluating trained agent --> load it
    if args.mode != "random_exploration":
        loaded_model = DQN.load(model_name, env=eval_env)
    else:
        loaded_model = None # No model to load for random

    obs, info = eval_env.reset()
    for episode in range(5):
        done = False
        while not done:
            if loaded_model:
                action, _states = loaded_model.predict(obs, deterministic=True)
            else: # Random agent
                action = eval_env.action_space.sample()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render()
            done = terminated or truncated
            if done:
                print(f"Episode {episode + 1} finished (mode: {args.mode})")
                obs, info = eval_env.reset()
    eval_env.close()

if __name__ == "__main__":
    main()