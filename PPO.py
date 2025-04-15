# Add comments later ...

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
BATCH_SIZE = 4                  # A misnomer; should be rather called num_envs
ROLLOUTS = 30
GAMMA = 0.99
ALPHA = 1e-3
VL_COEF = 0.5
NUM_EPOCHS = 2
LAMBDA = 0.97
NUM_TRANSITIONS = int(BATCH_SIZE * MAX_STEP)
NUM_MINI_BATCHES = 4
MINI_BATCH_SIZE = int(NUM_TRANSITIONS // NUM_MINI_BATCHES)
CLIP_COEF = 0.2

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

    def forward(self, state, action=None):                   
        logits = self.linear_relu_stack(state)
        probs = Categorical(logits=logits) 
        if not action:
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
    actions = torch.zeros((MAX_STEP, BATCH_SIZE), dtype=torch.long).to(device)      # Store as long or else accidental conversion possible; float 32 default; is another action_space dimension neede?
    dones = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)      
    rewards = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)   
    logprobs = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)   
    values = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)

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

            logprob, action = agent(next_obs)
            value = agent.value(obs[step]).flatten()              
            values[step] = value
            logprobs[step] = logprob     
            actions[step] = action                      # Needed later for recomputing probabilyt under new policy

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
        value_next_state = agent.value(next_obs).flatten()          # ! Needed as next state my be contuination or reset as we have arbitraty cut of of trajectories
        advantages = torch.zeros((MAX_STEP, BATCH_SIZE)).to(device)         # Why are we storin?
        generalized_advantage_estimate = 0

        ####### !!!!  I made serveral errors calculating this
        # e.g. I assumed that the value of the next step is 
        # always 0 in the REINFORCE and actor-critic versions, 
        # I used the mask for the current step tjo multiply by the
        # value of the next step.
        for step in reversed(range(MAX_STEP)):          # Can be outside epoch loop as stuff does not need to be recomputed
            if step == MAX_STEP - 1:
                value_next_state = value_next_state
                next_terminal = 1 - next_done
            else:
                value_next_state = values[step+1]
                next_terminal = 1 - dones[step+1]

            td_target = rewards[step] + GAMMA * value_next_state * next_terminal 
            td_error = td_target - value_state      ##### !!!!!!!! I initially used only the reward here, not td_target, which meant the model didn't learn anything; it got worse
            generalized_advantage_estimate = td_error + GAMMA * LAMBDA * generalized_advantage_estimate * next_terminal
            advantages[step] = generalized_advantage_estimate 
            returns = advantages + values           # Later used for calculating value loss; this deviates from the original PPO implementation

        # We now flatten the tensors to make mini-batching possible/easier
        # This means the temporal association between types state,action, etc
        # is destroyed. It doesn't seem to be needced.
        # The argument is that minibatching allows multiple updates to the thetas
        # per epoch - so is finer updating - also the mixing of experiences from differnt 
        # environments is allegedly helpful, plus easier to process all data.

        # TODO: Flatten tensors
        obs_flattened = obs.reshape((-1,) + envs.single_observation_space.shape)            # We are doing tuple math here; hence (-1, ); and 2 dimension needed to retain meaningful datat
        logprobs_flattened = logprobs.reshape(-1)
        actions_flattened = actions.reshape(-1)
        advantages_flattened = advantages.reshape(-1)
        returns_flattened = returns.reshape(-1)

        # The main poing what we can deanchor tuples seems to be that the temporal 
        # order is already baked in via the advantage calculation.  
        # BUT THESE THNINGS ARE ABSTRACT AND I NEED TO BETTER UNDERSTAND THEM
        batch_indices = np.arrange(NUM_TRANSITIONS)

        for epoch in NUM_EPOCHS:
            np.random.shuffle(batch_indices)
            
            for start in range(0, NUM_TRANSITIONS, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mini_batch_indices = batch_indices[start:end]       # Getch array of randomised batch indices

                new_logprobs, _ = agent(obs_flattened[mini_batch_indices], actions_flattened[mini_batch_indices])  
                new_values = agent.value(obs_flattened[mini_batch_indices]) 
                logratio = new_logprobs - logprobs_flattened[mini_batch_indices]
                ratio = logratio.exp()              # log(a/b) = log(a) - log(b); exponentiation inverse of log

                ### Loss calculations ###
                mini_batch_advantages = advantages_flattened[mini_batch_indices]
                unclipped_policy_loss = -ratio * mini_batch_advantages
                clipped_policy_loss = -torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * mini_batch_advantages
                policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()          # Using negative losses given what pytorch expects; taking mean as fomrula asks for expectation

                new_values = new_values.flatten()               # Needed as what comes out of critic has shape (x, 1)?
                value_loss = 0.5 * ((new_values - returns_flattened[mini_batch_indices])**2).mean()     # Taking expectation;
                loss = policy_loss + VL_COEF * value_loss             # Uses combined loss - is this smart? Why is this okay?

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Log loss
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        logprobs = []

    envs.close()
    writer.close()
