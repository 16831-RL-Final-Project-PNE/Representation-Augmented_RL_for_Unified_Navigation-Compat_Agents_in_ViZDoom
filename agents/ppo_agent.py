# agents/ppo_agent.py
from typing import Tuple, Dict
import numpy as np
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .nn_actor_critic import ActorCritic
from .nn_actor_critic_dinov3 import DinoV3ActorCritic
from .nn_actor_critic_dinov2 import DinoV2ActorCritic



class PPOAgent(nn.Module):
    """
    PPO agent wrapping an ActorCritic network.

    The RLTrainer interacts with this class ONLY through:
    - act(...)
    - get_value(...)
    - evaluate_actions(...)
    - update(...)         # PPO-specific training logic
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        feat_dim: int = 256,
        backbone: str = "cnn",
        freeze_backbone: bool = True,
        jepa_ckpt_path: str | None = None,
        jepa_partial_unfreeze: int = 0,
    ):
        """
        obs_shape: (frame_stack, 3, H, W) from DoomEnv.observation_shape
        n_actions: size of discrete action space

        backbone:
            "cnn"    -> use simple Conv encoder (ActorCritic)
            "dinov2" -> use DinoV2ActorCritic
            "dinov3" -> use DinoV3ActorCritic (requires HF gated access)
            HuggingFace model name for DINO backbone when using dinov2/dinov3.
        """
        super().__init__()
        frame_stack, c, h, w = obs_shape
        assert c == 3, f"Expected 3 channels per frame (RGB), got {c}"
        in_channels = frame_stack * c

        self.backbone = backbone.lower()
        self.n_actions = n_actions

        if self.backbone == "cnn":
            # Original CNN-based actor-critic
            self.ac = ActorCritic(
                in_channels=in_channels,
                n_actions=n_actions,
                feat_dim=feat_dim,
            )

            # ---- Load JEPA-pretrained encoder weights (if provided) ----
            if jepa_ckpt_path is not None:
                print(f"[PPOAgent] Loading JEPA encoder from: {jepa_ckpt_path}")
                ckpt = torch.load(jepa_ckpt_path, map_location="cpu", weights_only=True)

                if "encoder_state_dict" in ckpt:
                    enc_state = ckpt["encoder_state_dict"]
                elif "jepa_state_dict" in ckpt:
                    # get encoder from jepa_state_dict
                    raw_state = ckpt["jepa_state_dict"]
                    enc_state = {
                        k.replace("encoder.", ""): v
                        for k, v in raw_state.items()
                        if k.startswith("encoder.")
                    }
                else:
                    raise KeyError(
                        "JEPA checkpoint must contain 'encoder_state_dict' "
                        "or 'jepa_state_dict'."
                    )

                missing, unexpected = self.ac.enc.load_state_dict(enc_state, strict=False)
                if missing:
                    print(f"[PPOAgent] Warning: missing keys in encoder_state_dict: {missing}")
                if unexpected:
                    print(f"[PPOAgent] Warning: unexpected keys in encoder_state_dict: {unexpected}")

            # ---- Freeze / partial fine-tune encoder ----
            if freeze_backbone and (jepa_ckpt_path is not None):
                # exp A：all encoder frozen (only learning policy/value head)
                print("[PPOAgent] Freezing entire ConvEncoder (JEPA features frozen).")
                for p in self.ac.enc.conv.parameters():
                    p.requires_grad = False

            elif jepa_partial_unfreeze > 0 and (jepa_ckpt_path is not None):
                # exp B：first all freezing, and then undreeze last k conv layers + head
                print(f"[PPOAgent] Partially fine-tuning encoder: last {jepa_partial_unfreeze} conv layers + head.")
                for p in self.ac.enc.parameters():
                    p.requires_grad = False

                # Find out conv layers in ConvEncoder
                conv_modules = []
                for m in self.ac.enc.conv.modules():
                    if isinstance(m, nn.Conv2d):
                        conv_modules.append(m)

                # Unfreezing last k conv layers
                k = min(jepa_partial_unfreeze, len(conv_modules))
                for conv in conv_modules[-k:]:
                    for p in conv.parameters():
                        p.requires_grad = True

                # Unfreezing head (linear layer)
                for p in self.ac.enc.head.parameters():
                    p.requires_grad = True

        elif self.backbone == "dinov2":
            # DINOv2-based actor-critic (encoder is usually frozen)
            self.ac = DinoV2ActorCritic(
                in_channels=in_channels,
                n_actions=n_actions,
                model_name="facebook/dinov2-small",
                freeze_backbone=freeze_backbone,
                hidden_dim=feat_dim,  # optional small MLP head dim
            )

        elif self.backbone == "dinov3":
            # DINOv3-based actor-critic (requires gated access on HF)
            self.ac = DinoV3ActorCritic(
                in_channels=in_channels,
                n_actions=n_actions,
                model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                freeze_backbone=freeze_backbone,
                hidden_dim=feat_dim,
            )

        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Expected one of ['cnn', 'dinov2', 'dinov3']."
            )

    # --- core API used during rollout / evaluation ---
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        obs: (B, C, H, W)
        Returns: actions, log_probs, values (all tensors with batch dimension)
        """
        logits, value = self.ac(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, value

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, C, H, W)
        Returns: values (B,)
        """
        _, value = self.ac(obs)
        return value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Used during PPO update to recompute log_probs, entropy, and values
        under the current policy parameters.
        """
        logits, values = self.ac(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    # --- PPO-specific update logic, called from RLTrainer.update_policy() ---
    def update(self, buffer, optimizer: torch.optim.Optimizer, config) -> Dict[str, float]:
        """
        Run PPO updates using trajectories stored in `buffer`.

        The buffer is expected to provide:
        - .get_minibatches(batch_size) → yields dicts with keys:
          ["observations", "actions", "log_probs", "advantages", "returns"]

        Returns a dict of scalar logs:
        {
            "Loss_Policy": float,
            "Loss_Value": float,
            "Loss_Entropy": float,
        }
        """
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Outer loop: epochs, with tqdm progress bar
        for epoch_idx in trange(
            config.ppo_epochs,
            desc="PPO update",
            leave=False,
        ):
            for batch in buffer.get_minibatches(config.batch_size):
                obs_batch = batch["observations"]  # (B, C, H, W) already on correct device
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy, values = self.evaluate_actions(obs_batch, actions)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        logs = {
            "Loss_Policy": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "Loss_Value": float(np.mean(value_losses)) if value_losses else 0.0,
            "Loss_Entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }
        return logs