# agents/nn_actor_critic_dinov3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoModel


class DinoV3Encoder(nn.Module):
    """
    Wrap a DINOv3 vision backbone from HuggingFace and expose a frozen
    image encoder that maps (B, 3, H, W) -> (B, out_dim).

    We do NOT use AutoImageProcessor because the gated repo
    'facebook/dinov3-vits16-pretrain-lvd1689m' does not provide a
    preprocessor_config.json. Instead we implement simple ImageNet-style
    preprocessing by hand.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        out_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # trust_remote_code=True is important for DINOv3
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Figure out feature dim from config
        hidden_dim = getattr(self.backbone.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.backbone.config, "embed_dim", None)
        if hidden_dim is None:
            raise ValueError(
                "Could not infer hidden size from DINOv3 config "
                "(looked for 'hidden_size' and 'embed_dim')."
            )

        self.feat_dim = hidden_dim
        self.proj = nn.Linear(self.feat_dim, out_dim)

        # Typical ImageNet normalization (approximation for DINOv3)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        # Many ViT-style models use 224x224; fall back to this if missing
        img_size = getattr(self.backbone.config, "image_size", 224)
        if isinstance(img_size, int):
            self.image_size = (img_size, img_size)
        else:
            # in case it is already a tuple/list
            self.image_size = tuple(img_size)

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [0, 1] or uint8 [0, 255]
        returns: (B, out_dim)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Resize to model's native resolution
        x = F.interpolate(
            x,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # DINO-style models usually take 'pixel_values'=(B,3,H,W)
        outputs = self.backbone(pixel_values=x)

        # Get a single feature vector per image
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output  # (B, D)
        elif hasattr(outputs, "last_hidden_state"):
            # (B, N, D) tokens: use CLS token if present, else mean pool
            tokens = outputs.last_hidden_state
            if tokens.shape[1] >= 1:
                feat = tokens[:, 0]  # CLS
            else:
                feat = tokens.mean(dim=1)
        elif isinstance(outputs, torch.Tensor):
            feat = outputs
        else:
            raise RuntimeError("Unexpected outputs type from DINOv3 backbone.")

        feat = self.proj(feat)  # (B, out_dim)
        return feat


class DinoV3ActorCritic(nn.Module):
    """
    Actor-critic that uses a frozen DINOv3 encoder as visual backbone.
    We only fine-tune a small MLP on top.
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_actions: int = 12,
        hidden_dim: int = 256,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # We will use only the last RGB frame (3 channels) from the stack.
        assert in_channels % 3 == 0, "in_channels must be multiple of 3"
        self.n_frames = in_channels // 3

        self.encoder = DinoV3Encoder(
            model_name=model_name,
            out_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
        )

        # Small heads on top of frozen features
        self.pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.v = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W) where C = 3 * frame_stack (RGB stacks).
        We take only the most recent frame for DINOv3.
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Take the last RGB frame: shape (B, 3, H, W)
        last_rgb = x[:, -3:, :, :]

        feat = self.encoder(last_rgb)  # (B, feat_dim)
        logits = self.pi(feat)         # (B, n_actions)
        value = self.v(feat).squeeze(-1)  # (B,)
        return logits, value

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if deterministic:
            a = torch.argmax(dist.probs, dim=-1)
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value


if __name__ == "__main__":
    # Simple shape test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DinoV3ActorCritic(in_channels=12, n_actions=12).to(device)
    x = torch.randint(0, 256, (2, 12, 84, 84), dtype=torch.uint8, device=device)
    logits, value = net(x)
    print("logits:", logits.shape, "value:", value.shape)
