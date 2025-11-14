# configs/ppo_config.py
from dataclasses import dataclass


@dataclass
class PPOConfig:
    total_iterations: int = 100
    steps_per_iteration: int = 4096
    batch_size: int = 64
    learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    ppo_epochs: int = 4
    clip_coef: float = 0.2
    value_coef: float = 0.5
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