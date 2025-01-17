import gymnasium as gym
from stable_baselines3 import PPO
import panda_gym
import time

# Path to the fine-tuned model
FINE_TUNED_MODEL_PATH = "../ppo_robot_arm_fine_tuned.zip"

def run_fine_tuned_model(env_id="PandaReachDense-v3", steps=1000):
    """
    Run the environment using the fine-tuned PPO model.

    Args:
        env_id (str): The environment ID (default is PandaReach-v3).
        steps (int): Number of steps to run the environment.
    """
    print("Loading fine-tuned model...")
    try:
        env = gym.make(env_id, render_mode="human")

        model = PPO.load(FINE_TUNED_MODEL_PATH)

        obs, info = env.reset()

        print(f"Running environment: {env_id} for {steps} steps...")
        for _ in range(steps):
            time.sleep(0.3)
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            env.render()

            done = terminated or truncated

            if done:
                obs, info = env.reset()

        print("Environment run completed!")
    except Exception as e:
        print(f"Error during environment run: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    run_fine_tuned_model()
