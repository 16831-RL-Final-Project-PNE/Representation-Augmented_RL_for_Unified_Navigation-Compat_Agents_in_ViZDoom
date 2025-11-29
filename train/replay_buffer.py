# train/replay_buffer.py
from typing import Tuple, Dict, Iterator

import numpy as np
import torch


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores transitions and computes GAE advantages and returns.
    Observations are stored as (T, 3, H, W) uint8 arrays.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        obs_shape: (frame_stack, 3, H, W)
        """
        self.buffer_size = int(buffer_size)
        self.obs_shape = obs_shape
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=torch.uint8
        )
        self.actions = torch.zeros(self.buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32)

        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32)

        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.pos

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        obs: (frame_stack, 3, H, W) numpy array (uint8)
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("RolloutBuffer is full; call reset() before adding more.")

        self.observations[self.pos].copy_(torch.from_numpy(obs))
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """
        Compute GAE advantages and returns in-place.
        last_value: bootstrap value for the final state after the rollout.
        """
        last_value_t = torch.tensor(last_value, dtype=torch.float32)
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0.0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value_t
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_advantage = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            )
            advantages[t] = last_advantage

        self.advantages[: self.size] = advantages[: self.size]
        self.returns[: self.size] = self.advantages[: self.size] + self.values[: self.size]

    def get_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield minibatches as dictionaries.
        Observations are returned as float32 normalized to [0,1]
        with shape (B, C, H, W) where C = 3 * frame_stack.
        """
        inds = np.arange(self.size)
        np.random.shuffle(inds)

        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_idx = inds[start:end]

            obs_batch = self.observations[batch_idx].to(self.device)  # (B, T, 3, H, W)
            b, t, c, h, w = obs_batch.shape
            obs_batch = obs_batch.view(b, t * c, h, w).float() / 255.0

            yield {
                "observations": obs_batch,
                "actions": self.actions[batch_idx].to(self.device),
                "log_probs": self.log_probs[batch_idx].to(self.device),
                "advantages": self.advantages[batch_idx].to(self.device),
                "returns": self.returns[batch_idx].to(self.device),
                "values": self.values[batch_idx].to(self.device),
            }
