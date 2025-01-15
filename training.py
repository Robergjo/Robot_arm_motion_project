import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import panda_gym
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import time

def training(env, network, optimizer, ppo_loss):
    max_steps = 1000
    previous_distance = None
    # Träningsloop
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    
    for episode in range(10): 
        old_log_probs = log_probs
        print("For loop. Episode", episode)
        time.sleep(0.5)
        state, _ = env.reset()   
        # print(state)
        desired_goal = state["desired_goal"]
        state = state["observation"]
        # print(state)
        # print("-----------------")
        done = False
        step = 0

        while True:
            step += 1
            if step >= max_steps:
                print("Max steps reached")
                break
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, value = network.forward(state_tensor)
            # print(action,"--", value)
            # if episode <= 100:
            #     action = env.action_space.sample()
            #     if episode % 100 == 0:
            #         # print("Random Action", action)
            #         ac = action
            # else:
            # if step % 100 == 0 or step <= 20:
            #     print("Count", step)
            #     print("1:", action)
            action = action.detach().numpy()[0]
            
            action = np.clip(action, env.action_space.low, env.action_space.high)            

            next_state, reward, done, _, _ = env.step(action)
            achieved_goal = next_state["achieved_goal"]

            # Compute reward
            reward, current_distance = compute_distance_reward(achieved_goal, desired_goal, previous_distance)
            previous_distance = current_distance

            log_prob = -0.5 * ((action - action.mean()) ** 2).sum()
            
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            state = next_state["observation"]
    
            # Uppdatera nätverket
        for _ in range(epochs):
            # print("PPO Loss")
            policy_loss = ppo_loss.compute_loss(
                states, actions, rewards, values, log_probs, old_log_probs, values[1:], dones
            )
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        print(f"Episode {episode}, Total Reward: {np.sum(rewards)}")