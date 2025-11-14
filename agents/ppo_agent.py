# agents/ppo_agent.py
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .nn_actor_critic import ActorCritic


class PPOAgent(nn.Module):
    """
    PPO agent wrapper around an Actor-Critic network.
    Assumes stacked RGB frames as input.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int, int],
        n_actions: int,
        feat_dim: int = 256,
    ) -> None:
        """
        obs_shape: (frame_stack, 3, H, W)
        """
        super().__init__()
        if len(obs_shape) != 4:
            raise ValueError("obs_shape must be (frame_stack, 3, H, W)")

        frame_stack, channels, height, width = obs_shape
        if channels != 3:
            raise ValueError(f"Expected 3 channels (RGB) in obs_shape, got {channels}")

        in_channels = frame_stack * channels
        self.actor_critic = ActorCritic(
            in_channels=in_channels,
            n_actions=n_actions,
            feat_dim=feat_dim,
        )

    def forward(self, obs: torch.Tensor):
        """
        Forward through the underlying ActorCritic.
        obs: (B, C, H, W) float32 in [0,1] or [0,255].
        """
        return self.actor_critic(obs)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        obs: (B, C, H, W)
        Returns:
            actions: (B,)
            log_probs: (B,)
            values: (B,)
        """
        logits, values = self.actor_critic(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Evaluate log probabilities, entropy and values for given actions.
        Returns:
            log_probs: (B,)
            entropy:  (B,)
            values:   (B,)
        """
        logits, values = self.actor_critic(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for given observations.
        obs: (B, C, H, W)
        Returns:
            values: (B,)
        """
        _, values = self.actor_critic(obs)
        return values
