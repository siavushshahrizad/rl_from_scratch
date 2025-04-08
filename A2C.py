# This implementation uses the TD error instead of G;
# See the bare REINFORCE.py implementation for this

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
ROLLOUTS = 40
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

        self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            ) 

    def forward(self, state):
        logits = self.linear_relu_stack(state)
        probs = Categorical(logits=logits) 
        action = probs.sample() 
        return probs.log_prob(action), action 
    
    def value(self, state):
        return self.critic(state)


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
    # logprobs = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)   # Not needed for REINFORCE, only PPO
    logprobs = []

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
            # logprobs[step] = logprob                            # Only for PPO; breaks computational graph
            logprobs.append(logprob)

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
        policy_loss = torch.zeros(BATCH_SIZE).to(device) 
        value_loss = torch.zeros(BATCH_SIZE).to(device)
        value_next_state = torch.zeros(BATCH_SIZE).to(device)

        for step in reversed(range(MAX_STEP)):
            value_state = agent.value(obs[step])
            td_error = rewards[step] - value_state + GAMMA * value_next_state * (1 - dones[step])
            policy_loss += -logprobs[step] * td_error
            value_loss += 0.5 * ((rewards[step] - value_state)**2)           # Uses a squared sum per envrionment error - is this okay?
            value_next_state = value_state

        optimizer.zero_grad()
        policy_loss = policy_loss.mean()
        value_loss = value_loss.mean()
        loss = policy_loss + value_loss             # Uses combined loss - is this smart?
        loss.backward()
        optimizer.step()

        # Log loss
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        logprobs = []

    envs.close()
    writer.close()
