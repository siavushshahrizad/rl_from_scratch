import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn

# Hyperparameters
MAX_STEP = 200
BATCH_SIZE = 4
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

    def forward(self, state):
        logits = self.linear_relu_stack(state)
        probs = self.sfmax(logits)
        action = torch.argmax(probs, dim=-1)
        return probs, action 

    def sfmax(self, logits):
        numerator = torch.exp(logits)
        denominator = torch.sum(numerator, dim=-1, keepdim=True)      # Makes sure that probability scores are calculated for left and right per sample in batch 
        probs = numerator / denominator
        return probs

def make_env():
    return gym.make("CartPole-v1")

if __name__ == "__main__":

    # Setup up environment
    envs = gym.vector.SyncVectorEnv([make_env for _ in range(BATCH_SIZE)])
    
    # Create agent
    agent = Agent().to(device)

    # Storage
    rollout_rewards = torch.zeros((ROLLOUTS, BATCH_SIZE)).to(device)

    for rollout in range(ROLLOUTS):
        
        # Reset at beginning of rollouts
        total_reward = torch.zeros(BATCH_SIZE).to(device)
        next_obs, info = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        steps = 0

        dones = np.zeros(BATCH_SIZE) 

        print("loop about to run")
        while not np.all(dones) or steps < MAX_STEP:
            steps += 1
            obs_tensor = torch.Tensor(next_obs).to(device)
            with torch.no_grad():
                logits, action = agent(obs_tensor)
            
            action_numpy = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs.step(action_numpy)
        
            dones = np.logical_or(terminated, truncated)
            masks = 1 - dones
            masks = torch.tensor(masks, dtype=torch.float32).to(device)
            total_reward += torch.tensor(reward, dtype=torch.float32).to(device) * masks
            
        print("loop done")
        rollout_rewards[rollout] = total_reward

    envs.close()
    
    # Print results
    plt.plot(rollout_rewards) 
    plt.xlabel("Rollout")
    plt.ylabel("Total Reward")
    plt.show()
