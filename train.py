import torch
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper # For pixel observations
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, DummyVecEnv
from minigrid_env import GridEnv

def main():
    print("--- Setting up & training agent ---")

    # Establish environment
    env = GridEnv(render_mode="rgb_array")  
    env = ImgObsWrapper(env)  # Convert to pixel observations
    env = DummyVecEnv([lambda: env]) 
    env = VecMonitor(env)  # Monitor the environment for logging

    # RL agent instantiation
    model = DQN(
        "CnnPolicy",  
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Agent training
    print("Starting training...")
    model.learn(total_timesteps=100000)  
    print("Training finished!")

    # Training eval
    print(" --- Evaluating trained agent ---")
    eval_env = GridEnv(render_mode="human")
    eval_env = ImgObsWrapper(eval_env)  

