# agents/random_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical


class RandomAgent(nn.Module):
    """
    Stateless random policy with the same interface as PPOAgent.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.n_actions = int(n_actions)

    def forward(self, obs: torch.Tensor):
        """
        This is not used for inference; required for nn.Module.
        """
        raise NotImplementedError("RandomAgent does not support forward()")

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        obs: (B, C, H, W) or anything with batch dimension.
        Returns:
            actions:   (B,)
            log_probs: (B,)
            values:    (B,) (always zero)
        """
        batch_size = obs.shape[0]
        device = obs.device

        actions = torch.randint(
            low=0,
            high=self.n_actions,
            size=(batch_size,),
            device=device,
        )

        # Uniform policy => p = 1 / n_actions
        if self.n_actions > 0:
            log_p = -torch.log(torch.tensor(self.n_actions, dtype=torch.float32, device=device))
        else:
            log_p = torch.tensor(0.0, device=device)

        log_probs = torch.full((batch_size,), log_p, device=device)
        values = torch.zeros(batch_size, device=device)

        return actions, log_probs, values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        For API compatibility with PPOAgent.
        Returns:
            log_probs: (B,)
            entropy:   (B,)
            values:    (B,)
        """
        batch_size = actions.shape[0]
        device = actions.device

        if self.n_actions > 0:
            log_p = -torch.log(torch.tensor(self.n_actions, dtype=torch.float32, device=device))
            entropy_val = torch.log(torch.tensor(self.n_actions, dtype=torch.float32, device=device))
        else:
            log_p = torch.tensor(0.0, device=device)
            entropy_val = torch.tensor(0.0, device=device)

        log_probs = torch.full((batch_size,), log_p, device=device)
        entropy = torch.full((batch_size,), entropy_val, device=device)
        values = torch.zeros(batch_size, device=device)

        return log_probs, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Value estimate for given observations: always zero.
        """
        batch_size = obs.shape[0]
        device = obs.device
        return torch.zeros(batch_size, device=device)
