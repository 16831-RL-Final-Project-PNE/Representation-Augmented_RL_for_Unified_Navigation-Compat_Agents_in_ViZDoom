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
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value_t
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

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
                "dones": self.dones[batch_idx].to(self.device),
                "rewards": self.rewards[batch_idx].to(self.device),
            }

    def _sample_idx(self, L: int):
        """
        Same behavior as TransitionBuffer:
        - sample any contiguous sequence of length L
        - circular buffer aware
        - DOES NOT reject done-crossing sequences
        """
        if self.full:
            max_start = self.buffer_size
        else:
            max_start = self.size - L

        start = np.random.randint(0, max_start)
        idxs = np.arange(start, start + L) % self.buffer_size
        return idxs

    def _retrieve_sequence_batch(self, idxs, batch_size, L):
        """
        idxs: (B, L)
        return EXACTLY like TransitionBuffer:
           obs:  (L, B, *obs_shape)
           act:  (L, B)
           rew:  (L, B)
           term: (L, B)
        """
        vec = idxs.reshape(-1)

        obs  = self.observations[vec].reshape(batch_size, L, *self.obs_shape)
        b, l, t, c, h, w = obs.shape
        obs = obs.view(b, l * t, c, h, w).float() / 255.0
        act  = self.actions[vec].reshape(batch_size, L)
        rew  = self.rewards[vec].reshape(batch_size, L)
        term = self.dones[vec].reshape(batch_size, L)

        # RETURN WITH THE SAME (L,B,...) ORDER AS TransitionBuffer
        obs  = obs.permute(1, 0, 2, 3, 4, 5) if obs.ndim == 6 else obs.permute(1,0,2,3,4)
        act  = act.permute(1, 0)
        rew  = rew.permute(1, 0)
        term = term.permute(1, 0)

        return obs, act, rew, term

    def sample_sequences(self, seq_len: int, batch_size: int):
        """
        EXACT TransitionBuffer behavior:
        Returns sequences of shape:
            obs:  (T, B, *obs_shape)
            act:  (T, B)
            rew:  (T, B)
            term: (T, B)
        """
        L = seq_len + 1

        # exactly like TransitionBuffer
        idxs = np.asarray([self._sample_idx(L) for _ in range(batch_size)])

        obs, act, rew, term = self._retrieve_sequence_batch(idxs, batch_size, L)

        # SHIFT like TransitionBuffer
        obs  = obs[1:]     # (T,B,...)
        act  = act[:-1]    # (T,B,action_size)
        rew  = rew[:-1]    # (T,B)
        term = term[:-1]   # (T,B)

        return {
            "observations": obs.to(self.device).float(),  
            "actions":      act.to(self.device),
            "rewards":      rew.to(self.device),
            "dones":        term.to(self.device),
        }