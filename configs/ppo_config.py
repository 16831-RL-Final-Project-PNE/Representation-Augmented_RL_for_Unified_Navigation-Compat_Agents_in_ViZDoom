# configs/ppo_config.py
from dataclasses import dataclass


@dataclass
class PPOConfig:
    total_iterations: int = 200
    steps_per_iteration: int = 8192
    batch_size: int = 64
    learning_rate: float = 1e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    ppo_epochs: int = 4
    clip_coef: float = 0.1
    value_coef: float = 0.25
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    eval_episodes: int = 10
    eval_interval: int = 1
    eval_log_path: str = "./logs/eval_returns.npz"

    # TensorBoard log directory
    log_dir: str = "./logs/tb_ppo"

    # ---- NEW: model checkpoint options ----
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "ppo_basic"   # prefix
    save_every: int = 0                  # 0 = only final, >0 = also every N iterations

    # whether to use deterministic policy during evaluation
    eval_deterministic: bool = True

    # encoder model
    feat_dim: int = 256
    backbone: str = "cnn" # "cnn" / "dinov2" / "dinov3"
    freeze_backbone: bool = False

    # ---------------------------
    # RND exploration (optional)
    # ---------------------------
    # When False, RLTrainer behaves exactly as before (no intrinsic reward).
    use_rnd: bool = False

    # Reward mixing:
    # total_reward = rnd_ext_coef * extrinsic + rnd_int_coef * normalized_intrinsic
    rnd_int_coef: float = 1.0
    rnd_ext_coef: float = 1.0

    # EMA smoothing for std of intrinsic reward
    rnd_gamma: float = 0.99

    # RND optimizer hyperparameters
    rnd_lr: float = 1e-4
    rnd_weight_decay: float = 1e-4

    # RND training schedule per rollout
    rnd_batch_size: int = 256
    rnd_epochs: int = 1

    # Whether to linearly decay rnd_int_coef from its initial value to 0
    rnd_int_decay: bool = False