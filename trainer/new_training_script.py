import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

def main():
    # Envirom,ent
    env_id = "PandaPushDense-v3"
    log_dir = "./ppo_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap the environment to monitor åperformance
    env = make_vec_env(env_id, n_envs=1)
    env = Monitor(env, log_dir)

    # PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

    # Eval
    eval_env = make_vec_env(env_id, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ppo_best_model/",
        log_path="./ppo_eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Step 4: Train the model
    print("Starting training...")
    model.learn(total_timesteps=50000, callback=eval_callback)
    print("Training completed!")

    # Step 5: Save the model
    model.save("ppo_panda_push")
    print("Model saved to ppo_panda_push.zip")

    # Step 6: Load and test the model
    loaded_model = PPO.load("ppo_panda_push")
    obs = eval_env.reset()

    for _ in range(1000):
        action, _states = loaded_model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render()

if __name__ == "__main__":
    main()
