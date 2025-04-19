# TODO: Add annealing
# TODO: Add baseline
# TODO: Entropy regularisation
# TODO: Reward normalisation

import torch
import argparse
import gymnasium as gym
from torch import nn


# Default hyperparameters
NUM_ENVS = 1
ENV = "FrozenLake-v1"
ALPHA = 1e-3
GAMMA = 0.99
ROLLOUTS = 500

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN_IN),
                nn.ReLU(),
                nn.Linear(HIDDEN_IN, HIDDEN_OUT),
                nn.ReLU(),
                nn.Linear(HIDDEN_OUT, ACTION_DIM)
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


def main():
    args = parse_args()

if __name__ == "__main__":
    main()
