import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# Path to the saved model
SAVED_MODEL_PATH = "ppo_robot_arm.zip"
FINE_TUNED_MODEL_PATH = "ppo_robot_arm_fine_tuned.zip"

# Hyperparameters for fine-tuning
FINE_TUNE_PARAMS = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01
}

# Fine-tuning script
def fine_tune(env_id="PandaReachDense-v3", total_timesteps=100000):
    """
    Fine-tune a pre-trained PPO model on a specified environment.

    Args:
        env_id (str): The environment ID (default is PandaReachDense-v3).
        total_timesteps (int): Number of timesteps for fine-tuning.
    """
    # Create the environment and wrap with Monitor
    print(f"Creating environment: {env_id}")
    base_env = gym.make(env_id)
    env = make_vec_env(lambda: Monitor(base_env), n_envs=1)

    if not os.path.exists(SAVED_MODEL_PATH):
        raise FileNotFoundError(f"Saved model not found at {SAVED_MODEL_PATH}")

    print("Loading pre-trained model...")
    model = PPO.load(SAVED_MODEL_PATH, env=env, **FINE_TUNE_PARAMS)

    def custom_eval():
        print("Running custom evaluation...")
        eval_rewards = []
        for episode in range(5):
            obs = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
        print(f"Evaluation rewards: {eval_rewards}")
        print(f"Average reward: {sum(eval_rewards) / len(eval_rewards)}")

    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                                  log_path="./logs/",
                                  eval_freq=10000,
                                  deterministic=True,
                                  render=False)

    # Fine-tune the model
    print("Starting fine-tuning...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Custom evaluation
    custom_eval()

    # Save the fine-tuned modelö
    print(f"Saving fine-tuned model to {FINE_TUNED_MODEL_PATH}...")
    model.save(FINE_TUNED_MODEL_PATH)
    print("Fine-tuning completed!")

if __name__ == "__main__":
    fine_tune()

