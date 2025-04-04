import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn

# Hyperparameters
ROLLOUTS = 200

# This is a Mac implementation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, state, deterministic=True):
        logits = self.linear_relu_stack(state)
        if deterministic:
            action = int(torch.argmax(logits))
        return logits, action 


if __name__ == "__main__":

    # Setup up environment
    env = gym.make("CartPole-v1") 
    
    # Create agent
    agent = Agent().to(device)

    # Storage
    rollout_rewards = np.zeros(ROLLOUTS)

    for rollout in range(ROLLOUTS):
        
        # Reset at beginning of rollouts
        total_reward = 0
        next_obs, info = env.reset()

        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(next_obs).to(device)
                logits, action = agent(obs_tensor)

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rollout_rewards[rollout] = total_reward

    env.close()
    
    # Print results
    plt.plot(rollout_rewards) 
    plt.xlabel("Rollout")
    plt.ylabel("Total Reward")
    plt.show()
