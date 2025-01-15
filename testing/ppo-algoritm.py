import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PPOAgent:
    def __init__(self, env, network, optimizer, clip_epsilon=0.2, gamma=0.99, lambda_gae=0.95, 
                 update_epochs=10, batch_size=64):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        return advantages

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, value = self.network(state_tensor)

                action_np = action.detach().numpy()[0]
                action_clipped = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
                next_state, reward, done, _ = self.env.step(action_clipped)

                log_prob = -0.5 * ((action - action.mean()) ** 2).sum()

                states.append(state_tensor)
                actions.append(torch.FloatTensor(action_np))
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)

                state = next_state

            advantages = self.compute_gae(rewards, values, dones)
            advantages = torch.FloatTensor(advantages)
            returns = advantages + torch.FloatTensor(values)

            # Update the network
            self.update_network(states, actions, log_probs, returns, advantages)

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")

    def update_network(self, states, actions, old_log_probs, returns, advantages):
        dataset = list(zip(states, actions, old_log_probs, returns, advantages))

        for _ in range(self.update_epochs):
            np.random.shuffle(dataset)

            for batch_start in range(0, len(dataset), self.batch_size):
                batch = dataset[batch_start:batch_start + self.batch_size]
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = zip(*batch)

                batch_states = torch.cat(batch_states)
                batch_actions = torch.stack(batch_actions)
                batch_old_log_probs = torch.FloatTensor(batch_old_log_probs)
                batch_returns = torch.FloatTensor(batch_returns)
                batch_advantages = torch.FloatTensor(batch_advantages)

                # Compute new log_probs
                action_preds, value_preds = self.network(batch_states)
                log_probs = -0.5 * ((action_preds - action_preds.mean()) ** 2).sum(dim=1)

                # Compute policy ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value loss
                value_loss = (batch_returns - value_preds).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Update the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

if __name__ == "__main__":
    from robot_arm_env import RoboticArmEnv
    from network import ActorCriticNetwork

    env = RoboticArmEnv()
    state_dim = env.state_size
    action_dim = env.action_size

    network = ActorCriticNetwork(state_dim, action_dim)
    optimizer = optim.Adam(network.parameters(), lr=3e-4)

    agent = PPOAgent(env, network, optimizer)
    agent.train(num_episodes=100)