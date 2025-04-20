# File: VPG.py
# Created: 19 April 2025
# ------------------------------
# Summary
# ------------------------------
# This file runs the vanilla policy gradient (VPG) algorith.
# 


# TODO: Add annealing
# TODO: Add baseline
# TODO: Entropy regularisation
# TODO: Reward normalisation


import torch
import argparse
import platform
import gymnasium as gym
from torch import nn


# Default hyperparameters
NUM_ENVS = 1
ENV = "FrozenLake-v1"
ALPHA = 1e-3
GAMMA = 0.99
ROLLOUTS = 500
HIDDEN_IN = 128
HIDDEN_OUT = 128


class Network(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(observation_space, HIDDEN_IN),
                nn.ReLU(),
                nn.Linear(HIDDEN_IN, HIDDEN_OUT),
                nn.ReLU(),
                nn.Linear(HIDDEN_OUT, action_space)
                )

        def forward(self, state):
            logits = self.linear_relu_stack(state)
            probs = Categorical(logits=logits)
            action = probs.sample()
            return probs.log_prob(action), action


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla Policy Gradient Algorithm")

    parser.add_argument(
            "--num_envs", 
            type=int, 
            default=NUM_ENVS, 
            help="Number of parallel environments to run (default is 1)"
            )
    
    parser.add_argument(
            "--learning_rate",
            type=float, 
            default=ALPHA,
            help="Learning rate of the agent (default is 1e-3)"
            )

    parser.add_argument(
            "--gamma",
            type=float,
            default=GAMMA,
            help="Discount factor for rewards (default is 0.99)"
            )

    parser.add_argument(
            "--rollouts",
            type=int,
            default=ROLLOUTS,
            help="Number of rollouts (default is 500)"
            )

    return parser.parse_args()


def setup_device():
    system = platform.system()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def main():
    args = parse_args()
    device = setup_device()
    env = None
    agent = None


if __name__ == "__main__":
    main()
