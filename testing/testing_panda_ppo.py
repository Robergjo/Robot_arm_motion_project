import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import time

# Parallel environments
vec_env = make_vec_env("PandaPushDense-v3")
policy_kwargs = dict(net_arch=[128, 128])
model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="./ppo_PandaPush_tensorboard/", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=25000, progress_bar=True)
model.save("ppo_PandaPush")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_PandaPush")

obs = vec_env.reset()
step = 0
while True:
    step += 1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if step % 100 == 0:
        print(f"Step: {step}, Reward: {rewards}")
