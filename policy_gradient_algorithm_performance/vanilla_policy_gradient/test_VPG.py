import pytest
import torch
import gymnasium as gym
import sys

from VPG import parse_args, setup_device

class TestVPG:
    def test_parse_args(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "prog", "--num_envs", "3","--gamma", "0.99", "--learning_rate", "3e-4"
            ])
        args = parse_args()
        assert args.num_envs == 3 
        assert args.gamma == 0.99
        assert args.rollouts == 500
        assert args.learning_rate == 3e-4

    def test_cuda_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        device = setup_device()
        assert device.type == "cuda"

    def test_mps_availalbe(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        device = setup_device()
        assert device.type == "mps"

    def test_cpu_as_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        device = setup_device()
        assert device.type == "cpu"
