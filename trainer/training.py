from network import ActorCriticNetwork
from ppo_loss import PPOLoss
import torch
import numpy as np

def train(env):
    """
    Träna reinforcement learning-modellen med den angivna miljön.
    """
    # Hyperparameters
    lr = 1e-4
    gamma = 0.99
    clip_epsilon = 0.2
    lambda_gae = 0.95
    epochs = 10

    # Initiera nätverket, optimeraren och förlustklassen
    state_dim = env.state_size
    action_dim = env.action_size
    network = ActorCriticNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    ppo_loss = PPOLoss(clip_epsilon=clip_epsilon, gamma=gamma, lambda_=lambda_gae)

    # Träningsloop
    max_episodes = 1000
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, value = network(state_tensor)
            action = action.detach().numpy()[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Lägg till logik för att spara belöningar, tillstånd osv. för PPO

            state = next_state

        print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {episode_reward}")

        # Uppdatera nätverket här (lägg till PPO-logik)

