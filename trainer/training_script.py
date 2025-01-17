import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

LOG_DIR = "./ppo_logs/"
BEST_MODEL_DIR = "./ppo_best_model/"
MODEL_PATH = "ppo_robot_arm.zip"

def train_model(env_id="PandaReachDense-v3", total_timesteps=100000):
    """
    Train a PPO model on the specified environment.

    Args:
        env_id (str): The environment ID.
        total_timesteps (int): Number of timesteps for training.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])

    print("Initializing PPO model...")
    model = PPO("MultiInputPolicy", env, verbose=1)

    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed!")

    print(f"Saving the model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    print("Testing the trained model...")
    obs = env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    train_model()
