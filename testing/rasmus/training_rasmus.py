import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import panda_gym
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import time

def training(env, network, optimizer, ppo_loss):
    # Träningsloop
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    for episode in range(100000):
        time.sleep(0.5)
        state, _ = env.reset()   
        state = state["observation"]
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, value = network(state_tensor)
            # print(action,"--", value)
            if episode <= 100:
                action = env.action_space.sample()
                if episode % 100 == 0:
                    print("Random Action", action)
            else:
                action = action.detach().numpy()[0]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                if episode % 100 == 0:
                    print("Agent Action", action)

            next_state, reward, done, _, _ = env.step(action)

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
            policy_loss = ppo_loss.compute_loss(
                states, actions, rewards, values, log_probs, values[1:], dones
            )
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        print(f"Episode {episode}, Total Reward: {np.sum(rewards)}")