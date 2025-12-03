# agents/nn_actor_critic_dinov2.py
# Actor-Critic network using a frozen DINOv2 image encoder as visual backbone.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModel

# ---------------------------------------------------------------------
# DINOv2 encoder wrapper
# ---------------------------------------------------------------------

class DinoV2Encoder(nn.Module):
    """
    Wraps a pretrained DINOv2 vision transformer from HuggingFace.

    - Input:  (B, C, H, W) where C = 3 * frame_stack (e.g., 12 for 4 stacked RGB frames), C=3 in current setting.
    - Output: (B, feat_dim) feature embeddings.
    """

    def __init__(
        self,
        in_channels: int = 12,
        model_name: str = "facebook/dinov2-base",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        assert in_channels % 3 == 0, "in_channels must be a multiple of 3 (RGB frames)."
        self.in_channels = in_channels

        # Load pretrained DINOv2 model + processor
        self.model = AutoModel.from_pretrained(model_name)

        # Figure out the feature dimension from the backbone's config
        # For DINOv2 ViT models, it's usually called hidden_size
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            # Fallback: check projection dimension or last_hidden_state size
            hidden_size = getattr(self.model.config, "projection_dim", None)
            if hidden_size is None:
                raise ValueError(
                    "Could not infer feature dimension from DINOv2 model config."
                )
        self.feat_dim = int(hidden_size)

        # Typical ImageNet normalization (same as DINOv3)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        # input image size (fallback 224x224 if not provided)
        #img_size = getattr(self.model.config, "image_size", 224)
        # For ViT-S/14, 112x112 -> 8x8 tokens instead of 16x16 at 224x224.
        self.image_size = (112, 112)

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) with C = 3 * num_frames, uint8 or float in [0, 1]/[0, 255], num_frames=1 in current setting.
        Returns:
            feats: (B, feat_dim)
        """
        b, c, h, w = x.shape
        assert c % 3 == 0
        # only using last frame to speed up

        if x.dtype == torch.uint8:
            x = x.float() / 255.0 # (B, 3, H, W)

        # resize to expected resolution
        x = F.interpolate(
            x,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        # normalize
        img = (x - self.pixel_mean) / self.pixel_std

        # DINOv2 forward: pixel_values=(B,3,H,W)
        device = next(self.model.parameters()).device
        img = img.to(device)

        outputs = self.model(pixel_values=img)
        # For ViT-style models, CLS token is at index 0: (B*T, seq_len, hidden)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output          # (B, D)
        else:
            last_hidden = outputs.last_hidden_state  # (B*1, L, D)
            feats = last_hidden[:, 0, :]             # CLS token (B*1, D)

        return feats


# ---------------------------------------------------------------------
# Actor-Critic with DinoV2Encoder backbone
# ---------------------------------------------------------------------

class DinoV2ActorCritic(nn.Module):
    """
    Actor-Critic network that uses a frozen DINOv2 encoder as the visual backbone.

    - Encoder: DinoV2Encoder -> (B, feat_dim)
    - Policy head: MLP -> logits over actions
    - Value head: MLP -> scalar value
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_actions: int = 12,
        model_name: str = "facebook/dinov2-base",
        freeze_backbone: bool = True,
        hidden_dim: int | None = 256,
    ) -> None:
        super().__init__()
        self.encoder = DinoV2Encoder(
            in_channels=in_channels,
            model_name=model_name,
            freeze=freeze_backbone,
        )

        feat_dim = self.encoder.feat_dim

        # Optionally add a small MLP bottleneck on top of DINOv2 features
        proj_dim = hidden_dim if hidden_dim is not None else feat_dim
        if hidden_dim is not None:
            self.project = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.project = nn.Identity()

        # Policy & value heads
        self.pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, n_actions),
        )
        self.v = nn.Sequential(
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W), same as env observation after stacking (C = 3 * frame_stack).
        Strategy: use only the last RGB frame (C-3:C) as input to DINOv2 for speed.
        Returns:
            logits: (B, n_actions)
            value:  (B,)
        """
        device = next(self.parameters()).device
        x = x.to(device)

        # Take the last RGB frame: shape (B, 3, H, W)
        #last_rgb = x[:, -3:, :, :]
        #feats = self.encoder(last_rgb)      # (B, feat_dim)
        #h = self.project(feats)      # (B, proj_dim)

        B, C, H, W = x.shape
        T = C // 3
        frames = x.view(B, T, 3, H, W)  # (B, T, 3, H, W)

        feats_list = []
        for t in range(T):
            feats_list.append(self.encoder(frames[:, t]))  # each (B, feat_dim)

        feats = torch.stack(feats_list, dim=1).mean(dim=1)  # (B, feat_dim)
        h = self.project(feats)      # (B, proj_dim)
        logits = self.pi(h)          # (B, n_actions)
        value = self.v(h).squeeze(-1)  # (B,)
        return logits, value

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False):
        """
        x: (B, C, H, W)
        Returns:
            actions: (B,)
            log_probs: (B,)
            values: (B,)
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, value

    @torch.no_grad()
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility used in RLTrainer for bootstrapping.
        """
        _, value = self.forward(x)
        return value


if __name__ == "__main__":
    # Quick shape check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DinoV2ActorCritic(
        in_channels=12,
        n_actions=12,
        model_name="facebook/dinov2-base",
        freeze_backbone=True,
    ).to(device)

    x = torch.randint(0, 256, (2, 12, 84, 84), dtype=torch.uint8).to(device)
    logits, value = net(x)
    print("logits:", logits.shape, "value:", value.shape)  # (2,12), (2,)
