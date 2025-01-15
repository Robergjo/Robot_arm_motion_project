import torch

class PPOLoss:
    def __init__(self, clip_epsilon=0.2, gamma=0.99, lambda_=0.95):
        """
        Initiering av PPO-parametrar.
        """
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute_loss(self, states, actions, rewards, values, log_probs, old_log_probs, next_values, dones):
        """
        Beräkning av PPO-förlust.

        Args:
            states (Tensor): Nuvarande tillstånd.
            actions (Tensor): Utförda actions.
            rewards (Tensor): Erhållna belöningar.
            values (Tensor): Kritiker-värden från nätverket.
            log_probs (Tensor): Log-probabiliteter från den aktuella policyn.
            old_log_probs (Tensor): Log-probabiliteter från den tidigare policyn.
            next_values (Tensor): Kritiker-värden för nästa tillstånd.
            dones (Tensor): Flagga för episodslut.

        Returns:
            Tensor: Kombinerad förlust (policy + value).
        """
        # Generalized Advantage Estimation (GAE)
        advantages = []
        gae = 0
        dones = torch.tensor(dones, dtype=torch.float32).to(states.device)
        next_values = torch.cat([next_values, torch.zeros(1, device=states.device)])  # Lägger till sista värdet
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, device=states.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalisering

        # Returns (mål för kritiker)
        returns = advantages + values

        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)  # Viktning av policyförändring
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = (returns - values).pow(2).mean()

        # Entropy bonus för utforskning
        entropy_loss = -torch.mean(-log_probs * torch.exp(log_probs))

        # Kombinerad förlust
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
