import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


# Hyperparameters
MAX_STEP = 500
BATCH_SIZE = 4
ROLLOUTS = 4 
GAMMA = 0.99
ALPHA = 1e-3

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
        probs = Categorical(logits=logits) 
        action = probs.sample() 
        return probs.log_prob(action), action 

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
    optimizer = torch.optim.Adam(agent.parameters(), lr=ALPHA)    

    # Storage
    obs = torch.zeros((MAX_STEP, BATCH_SIZE) + envs.single_observation_space.shape).to(device)
    dones = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)      # Should this even be set up here?
    rewards = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)
    logprobs = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)   # Not needed for REINFORCE, only PPO

    for rollout in range(ROLLOUTS):
        
        ####### DATA COLLECTION #######
        #                             #
        ###############################

        # Reset at beginning of rollouts
        next_obs, info = envs.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        next_done = torch.zeros(BATCH_SIZE).to(device)

        for step in range(MAX_STEP):
            global_step += BATCH_SIZE
            obs[step] = next_obs
            dones[step] = next_done

            # with torch.no_grad():
            logprob, action = agent(next_obs)
            logprobs[step] = logprob                            # Only for PPO; breaks computational graph

            action_numpy = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs.step(action_numpy)

            next_done = np.logical_or(terminated, truncated) 
            next_done = torch.tensor(next_done, dtype=torch.float32).to(device)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            if "episode" in info:                           # The info object seems unique to each in environment and seems to differ between people's code. So need to check if this works
                returns = info["episode"]["r"] 
                for env_idx, env_return in enumerate(returns):
                    writer.add_scalar(f"charts/env_{env_idx}/episodic_return", env_return, global_step)         # This logs to Tensorboard; need to run server; and visit localhost to see results
        
        #######    TRAINING     #######
        #                             #
        ###############################
        loss = torch.zeros(BATCH_SIZE).to(device) 
        discounted_return = torch.zeros(BATCH_SIZE).to(device)

        for step in reversed(range(MAX_STEP)):
            discounted_return = rewards[step] + GAMMA * discounted_return * (1 - dones[step])
            # state_at_this_step = obs[step]
            # recomputed_log, _ = agent(state_at_this_step)
            # loss += -recomputed_log * discounted_return
            loss += -logprobs[step] * discounted_return
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    envs.close()
    writer.close()
