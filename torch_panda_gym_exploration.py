import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import panda_gym
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import time


# Environment
# env = gym.make('PandaReach-v3', render_mode="rgb_array")

env = gym.make("PandaPush-v3", render_mode="human")

# Lägg till video wrapper
# env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: e % 10 == 0)

# print("Observation Space:", env.observation_space)
# print("Action Space:", env.action_space)


# Build actor-critic network
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions scaled to [-1, 1]
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(256, 1)  # Outputs a single value
        )
    
    def forward(self, state):
        shared = self.shared(state)
        action = self.actor(shared)
        value = self.critic(shared)
        return action, value

# PPO loss class for training
class PPOLoss:
    def __init__(self, clip_epsilon=0.2, gamma=0.99, lambda_=0.95):
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute_loss(self, states, actions, rewards, values, log_probs, next_values, dones):
        # Discounted returns
        returns = []
        discounted_return = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            discounted_return = r + self.gamma * discounted_return * (1 - d)
            returns.insert(0, discounted_return)
        
        returns = torch.tensor(returns).to(states.device)
        advantages = returns - values.detach()

        # Policy loss
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = (returns - values).pow(2).mean()

        return policy_loss + 0.5 * value_loss



# Hyperparameters
gamma = 0.99
clip_epsilon = 0.2
lambda_gae = 0.95
lr = 3e-4
epochs = 10
batch_size = 64

# Miljö och nätverk
# print(env.observation_space)
observation, _ = env.reset()
state_dim = observation["observation"].shape[0]
action_dim = env.action_space.shape[0]
# print("-----------------")
# print(observation["observation"].shape)
network = ActorCriticNetwork(state_dim, action_dim).to(torch.device("cpu"))
optimizer = optim.Adam(network.parameters(), lr=lr)
ppo_loss = PPOLoss(clip_epsilon=clip_epsilon, gamma=gamma, lambda_=lambda_gae)

# Träningsloop
for episode in range(100000):
    time.sleep(0.5)
    state, _ = env.reset()   
    state = state["observation"]
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, value = network(state_tensor)
        # print(action,"--", value)
        if episode <= 10000:
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

