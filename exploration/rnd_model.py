# exploration/rnd_model.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init

from agents.nn_actor_critic import ConvEncoder


class RNDModel(nn.Module):
    """
    Random Network Distillation model.

    - target: fixed random ConvEncoder
    - predictor: trainable ConvEncoder
    - intrinsic reward: per-state prediction error between predictor and target

    This is used purely for exploration; PPO still has a single value function
    and policy. RND only reshapes the reward signal.
    """

    def __init__(
        self,
        obs_shape,
        output_size: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ) -> None:
        """
        obs_shape: (C, H, W) for flattened RGB stack, where C = 3 * frame_stack
        """
        super().__init__()
        in_channels, h, w = obs_shape
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Both encoders map (B, C, H, W) -> (B, output_size)
        self.target = ConvEncoder(in_channels=in_channels, feat_dim=output_size)
        self.predictor = ConvEncoder(in_channels=in_channels, feat_dim=output_size)

        # Per-dimension MSE; we reduce over feature dim manually
        self.loss_fn = nn.MSELoss(reduction="none")

        # Warm up LazyLinear inside both encoders on CPU so that their
        # parameters are fully materialized and no longer UninitializedParameter.
        self._warmup_lazy_layers(obs_shape=(in_channels, h, w))

        # Apply different initialization to target and predictor
        self._init_target(self.target)
        self._init_predictor(self.predictor)

        # Target network is fixed
        for p in self.target.parameters():
            p.requires_grad = False

        # Now move everything to the desired device
        self.to(self.device)

        # AdamW for the predictor only
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _warmup_lazy_layers(self, obs_shape) -> None:
        """
        Run a dummy forward pass through both encoders so that any LazyLinear
        layers are materialized before we touch .weight / .bias.

        This is done on CPU; after this we will move the whole module to
        self.device.
        """
        c, h, w = obs_shape
        dummy = torch.zeros(1, c, h, w, dtype=torch.float32)
        with torch.no_grad():
            _ = self.target(dummy)
            _ = self.predictor(dummy)

    def _init_target(self, module: nn.Module) -> None:
        """
        Initialization for the fixed random target network.
        Example: uniform initialization.
        """
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.1, b=0.1)

    def _init_predictor(self, module: nn.Module) -> None:
        """
        Initialization for the trainable predictor network.
        Example: normal initialization with a different distribution
        from the target, so the initial prediction error is not tiny.
        """
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _mse_per_sample(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, D)
        returns: (B,) mean squared error per sample
        """
        loss_per_dim = self.loss_fn(pred, target)  # (B, D)
        return loss_per_dim.mean(dim=1)            # (B,)

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward (prediction error) for a batch of observations.

        obs: (B, C, H, W) tensor in [0, 1] or uint8 [0, 255]
        returns: (B,) tensor of scalar intrinsic rewards
        """
        obs = obs.to(self.device)
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        target = self.target(obs)   # (B, D), no grad
        pred = self.predictor(obs)  # (B, D)
        mse_per_sample = self._mse_per_sample(pred, target)
        return mse_per_sample       # (B,)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Alias for compute_intrinsic_reward to keep nn.Module semantics.
        """
        return self.compute_intrinsic_reward(obs)

    def update(
        self,
        obs: torch.Tensor,
        batch_size: int = 256,
        epochs: int = 1,
    ) -> float:
        """
        Train the predictor network to match the fixed target network.

        obs: (N, C, H, W) tensor in [0, 1] or uint8 [0, 255]
        returns: average training loss over all samples (float)
        """
        if obs.numel() == 0:
            return 0.0

        obs = obs.to(self.device)
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        dataset = TensorDataset(obs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        total_count = 0

        self.train()
        for _ in range(epochs):
            for (batch_obs,) in loader:
                batch_obs = batch_obs.to(self.device)

                with torch.no_grad():
                    target = self.target(batch_obs)  # (B, D)

                pred = self.predictor(batch_obs)    # (B, D)
                loss_per_sample = self._mse_per_sample(pred, target)  # (B,)
                loss = loss_per_sample.mean()       # scalar

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_obs.size(0)
                total_count += batch_obs.size(0)

        self.eval()

        if total_count == 0:
            return 0.0
        return total_loss / float(total_count)
