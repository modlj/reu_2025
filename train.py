import torch
import gymnasium as gym
from gymnasium import spaces
from minigrid_env import GridEnv
from minigrid.wrappers import ImgObsWrapper # For pixel observations
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim), 
            torch.nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def main():
    print("--- Setting up & training agent ---")

    # Establish environment
    env = GridEnv(render_mode="rgb_array")  
    env = ImgObsWrapper(env)  # Convert to pixel observations
    env = DummyVecEnv([lambda: env]) 
    env = VecMonitor(env)  

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
        policy_kwargs=dict(
            features_extractor_class=GridCNN,  
            features_extractor_kwargs=dict(features_dim=256)
        )
    )

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



