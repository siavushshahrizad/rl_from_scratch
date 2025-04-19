import pytest
import torch
import gymnasium as gym
import sys

from VPG import parse_args

class TestVPG:
    def test_parser(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "prog", "--num_envs", "3","--gamma", "0.99", "--learning_rate", "3e-4"
            ])
        args = parse_args()
        assert args.num_envs == 3 
        assert args.gamma == 0.99
        assert args.rollouts == 500
        assert args.learning_rate == 3e-4
        
