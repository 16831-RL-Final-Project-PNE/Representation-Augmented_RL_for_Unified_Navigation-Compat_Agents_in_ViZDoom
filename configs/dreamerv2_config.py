# configs/dreamerv2_config.py
from dataclasses import dataclass, field

@dataclass
class DreamerV2Config:
    total_iterations: int = 200
    steps_per_iteration: int = 8192
    learning_rate: dict[str, float] = field(default_factory=lambda: {'world_model': 8e-5, 'actor_model': 3e-5, 'value_model': 1e-4})

    gamma: float = 0.995
    gae_lambda: float = 0.95
    
    epochs: int = 4

    eval_episodes: int = 10
    eval_interval: int = 1
    eval_log_path: str = "./logs/eval_returns.npz"
    # whether to use deterministic policy during evaluation
    eval_deterministic: bool = False

    # TensorBoard log directory
    log_dir: str = "./logs/tb_dreamerv2"

    # ---- NEW: model checkpoint options ----
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "dreamerv2_basic"   # prefix
    save_every: int = 0                  # 0 = only final, >0 = also every N iterations

    # DreamerV2 Model Config
    embedding_size: int = 512
    batch_size: int = 32
    grad_clip_norm: float = 10.0

    use_dino_v3: bool = False
    dino_v3_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dino_v3_freeze_backbone: bool = True

    use_slow_target: bool = True
    slow_target_fraction: float = 0.05

    rssm_deter_size: int = 512
    rssm_category_size: int = 20
    rssm_class_size: int = 20
    rssm_node_size: int = 256
    rssm_seq_len: int = 50
    rssm_horizon: int = 10

    kl_use_kl_balance: bool = True
    kl_balance_scale: float = 0.8
    kl_use_free_nats: bool = True
    kl_free_nats: float = 3.0

    actor_grad: str = 'dynamics' # 'reinforce' or 'dynamics'
    actor_entropy_scale: float = 1e-2
    actor_layers: int = 4
    actor_node_size: int = 256

    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 35000.0

    kl_loss_scale: float = 1.0
    discount_loss_scale: float = 1.0
    obs_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0

    actor_with_discount: bool = False
    reward_scale: float = 10.0