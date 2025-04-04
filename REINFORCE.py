import torch
import gymnasium as gym
from torch import nn

# Hyperparameters
STEPS = 200
ROLLOUTS = 200

# This is a Mac implementation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state):
        logits = self.linear_relu_stack(state)
        return logits


if __name__ == "__main__":

    # Setup up environment
    env = gym.make("CartPole-v1") 
    next_obs, info = env.reset()
    
    # Create agent
    agent = Agent().to(device)

    # Storage
    actions = []
    rewards = []
    observations = []

    for step in range(STEPS):
        observations.append(next_obs)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(int(action))

        if terminated or truncated:
            next_obs, info = env.reset()

    env.close()
    print(rewards)
    print(actions)

