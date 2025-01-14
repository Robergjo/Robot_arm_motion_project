import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import panda_gym
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import time
from training import training



# Environment
# env = gym.make('PandaReach-v3', render_mode="rgb_array")

env = gym.make("PandaReachDense-v3", max_episode_steps=128, render_mode="human")

# Lägg till video wrapper
# env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: e % 10 == 0)

# print("Observation Space:", env.observation_space)
# print("Action Space:", env.action_space)


# Build actor-critic network
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        shared = self.shared(state)
        action = self.actor(shared)
        value = self.critic(shared)
        return action, value

# PPO loss class for training
# class PPOLoss:
#     def __init__(self, clip_epsilon=0.2, gamma=0.99, lambda_=0.95):
#         self.clip_epsilon = clip_epsilon
#         self.gamma = gamma
#         self.lambda_ = lambda_

#     def compute_loss(self, states, actions, rewards, values, log_probs, next_values, dones):
#         # Discounted returns
#         returns = []
#         discounted_return = 0
#         for r, d in zip(reversed(rewards), reversed(dones)):
#             discounted_return = r + self.gamma * discounted_return * (1 - d)
#             returns.insert(0, discounted_return)
        
#         returns = torch.tensor(returns).to(states.device)
#         advantages = returns - values.detach()

#         # Policy loss
#         ratio = torch.exp(log_probs - log_probs.detach())
#         clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
#         policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

#         # Value loss
#         value_loss = (returns - values).pow(2).mean()

#         return policy_loss + 0.5 * value_loss


class PPOLoss:
    def __init__(self, clip_epsilon=0.2, gamma=0.99, lambda_=0.95):
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute_loss(self, states, actions, rewards, values, log_probs, old_log_probs, next_values, dones):
        """
        Compute the PPO loss.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions taken.
            rewards (Tensor): Rewards received.
            values (Tensor): Value function outputs.
            log_probs (Tensor): Log probabilities of actions from the current policy.
            old_log_probs (Tensor): Log probabilities of actions from the old policy.
            next_values (Tensor): Value function outputs for the next states.
            dones (Tensor): Done flags for episodes.

        Returns:
            Tensor: Combined loss (policy + value).
        """
        # Compute Generalized Advantage Estimation (GAE)
        advantages = []
        gae = 0
        dones = torch.tensor(dones, requires_grad=True, dtype=torch.float32).to(torch.device("cpu"))
        next_values.append(0)  # Handle last step for length mismatch
        print(len(rewards))
        for i in reversed(range(len(rewards))):
            # print("--------debug--------")
            # print("Reward", len(rewards), "next values", len(next_values),"dones",  len(dones),"values", len(values))
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, requires_grad=True).to(torch.device("cpu"))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

        # Compute returns (target values for the critic)
        
        values = torch.tensor(values, requires_grad=True).to(torch.device("cpu"))
        print("Advantages", advantages.dtype, "Values", values.dtype)
        returns = advantages + values

        # Policy loss
        # Convert to tensor
        log_probs = torch.tensor(log_probs, requires_grad=True).to(torch.device("cpu"))
        old_log_probs = torch.tensor(old_log_probs, requires_grad=True).to(torch.device("cpu"))
        print("Log probs", log_probs, "Old log probs", old_log_probs)

        ratio = torch.exp(log_probs - old_log_probs)  # Importance sampling ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = (returns - values).pow(2).mean()

        # Entropy bonus (optional, to encourage exploration)
        entropy_loss = -torch.mean(-log_probs * torch.exp(log_probs))

        print("Loss:",policy_loss + 0.5 * value_loss - 0.01 * entropy_loss)
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss




# Hyperparameters
gamma = 0.99
clip_epsilon = 0.2
lambda_gae = 0.95
lr = 2e-4
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

# Start training loop
# training(env=env, network=network, optimizer=optimizer, ppo_loss=ppo_loss)

# Träningsloop


def training(env, network, optimizer, ppo_loss):
    max_steps = 128
    previous_distance = None
    # Träningsloop
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    
    for episode in range(10): # Episode = when robot has reached the goal
        old_log_probs = log_probs
        print("For loop. Episode", episode)
        time.sleep(0.5)
        state, _ = env.reset()   
        print(state)
        desired_goal = state["desired_goal"]
        state = state["observation"]
        print(state)
        print("-----------------")
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
            if step % 100 == 0 or step <= 20:
                print("Count", step)
                print("1:", action)
            action = action.detach().numpy()[0]
            if step % 100 == 0:
                print("2:", action)
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            if step % 500 == 0 or step <= 20:
                print("After clip", action)
            

            next_state, reward, done, _, _ = env.step(action)
            achieved_goal = next_state["achieved_goal"]

            # Compute reward
            reward, current_distance = compute_distance_reward(achieved_goal, desired_goal, previous_distance)
            previous_distance = current_distance

            # if step % 500 == 0 or step <= 20:
            #     print(env.step(action))
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
            # states, actions, rewards, values, log_probs, old_log_probs, next_values, done
            # print("PPO done, now optimizer")
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        print(f"Episode {episode}, Total Reward: {np.sum(rewards)}")

def compute_distance_reward(achieved_goal, desired_goal, previous_distance=None):
    """
    Compute a reward based on whether the achieved goal is getting closer to the desired goal.

    Args:
        achieved_goal (numpy.array): The current position of the achieved goal (e.g., the cube's position).
        desired_goal (numpy.array): The target position of the goal.
        previous_distance (float, optional): The distance from the previous step. Default is None.

    Returns:
        reward (float): The computed reward.
        current_distance (float): The distance between the achieved goal and desired goal.
    """
    # Calculate the current distance between achieved and desired goal
    current_distance = np.linalg.norm(achieved_goal - desired_goal)

    # If previous distance is provided, reward progress
    if previous_distance is not None:
        # Reward for getting closer, penalize for moving away
        reward = (previous_distance - current_distance) * 10  # Scale for better gradients
    else:
        # Initial step or no prior distance: neutral reward
        reward = 0.0

    # Bonus reward for being very close to the desired goal
    if current_distance < 0.1:  # Threshold for "close enough"
        reward += 5.0

    return reward, current_distance

training(env=env, network=network, optimizer=optimizer, ppo_loss=ppo_loss)