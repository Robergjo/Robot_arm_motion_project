import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make("PandaReach-v3", 
               render_mode="human", 
               renderer="OpenGL",
               render_width=480,
               render_height=480,
               render_target_position=[0.2, 0, 0],
               render_distance=1.0,
               render_yaw=90,
               render_pitch=-70,
               render_roll=0,
)

env.reset()
env = DummyVecEnv([lambda: env])
image = env.render()

model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()


