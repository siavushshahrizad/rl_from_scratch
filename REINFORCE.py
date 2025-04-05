import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
MAX_STEP = 500
BATCH_SIZE = 4
ROLLOUTS = 1

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

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        logits = self.linear_relu_stack(state)
        probs = self.softmax(logits)
        action = torch.argmax(probs, dim=-1)
        return probs, action 

def make_env():
    def thunk():
        env = gym.make("CartPole-v1") 
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        return env
    return thunk

if __name__ == "__main__":

    # Track outcomes with Tensorboard
    timestamp = time.time()
    writer = SummaryWriter(f"runs/cartpole_{timestamp}")            # Needed for loging reward per episodce in Tensorboard
    global_step = 0

    # Setup up environment
    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(BATCH_SIZE)])
    
    # Create agent
    agent = Agent().to(device)

    # Storage
    returns_per_env = [[] for _ in range(BATCH_SIZE)]
    
    for rollout in range(ROLLOUTS):
        
        # Reset at beginning of rollouts
        next_obs, info = envs.reset()

        dones = np.zeros(BATCH_SIZE) 

        for step in range(MAX_STEP):
            global_step += BATCH_SIZE

            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, action = agent(next_obs)
            
            action_numpy = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs.step(action_numpy)
        
            dones = np.logical_or(terminated, truncated) 

            if "episode" in info:                           # The info object seems unique to each in environment and seems to differ between people's code. So need to check if this works
                returns = info["episode"]["r"] 
                for env_idx, env_return in enumerate(returns):
                    writer.add_scalar(f"charts/env_{env_idx}/episodic_return", env_return, global_step)         # This logs to Tensorboard; need to run server; and visit localhost to see results

    envs.close()
    writer.close()
