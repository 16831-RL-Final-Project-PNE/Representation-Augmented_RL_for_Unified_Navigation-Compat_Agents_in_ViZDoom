# configs/dreamerv2_config.py
from dataclasses import dataclass, field

@dataclass
class DreamerV2Config:
    total_iterations: int = 200
    steps_per_iteration: int = 8192
    batch_size: int = 64
    learning_rate: dict[str, float] = field(default_factory=lambda: {'world_model': 2e-4, 'actor_model': 4e-5, 'value_model': 1e-4})

    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    epochs: int = 4
    collect_intervals: int = 5

    eval_episodes: int = 10
    eval_interval: int = 1
    eval_log_path: str = "./logs/eval_returns.npz"
    # whether to use deterministic policy during evaluation
    eval_deterministic: bool = True

    # TensorBoard log directory
    log_dir: str = "./logs/tb_dreamerv2"

    # ---- NEW: model checkpoint options ----
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "dreamerv2_basic"   # prefix
    save_every: int = 0                  # 0 = only final, >0 = also every N iterations

    # DreamerV2 Model Config
    embedding_size: int = 256
    batch_size: int = 50
    horizon: int = 10
    grad_clip_norm: float = 100.0

    use_dino_v3: bool = True
    dino_v3_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dino_v3_freeze_backbone: bool = True

    rssm_deter_size: int = 200
    rssm_category_size: int = 20
    rssm_class_size: int = 20
    rssm_node_size: int = 100
    rssm_seq_len: int = 10

    kl_use_kl_balance: bool = True
    kl_balance_scale: float = 0.8
    kl_use_free_nats: bool = False
    kl_free_nats: float = 0.0

    actor_grad: str = 'reinforce' # 'reinforce' or 'dynamics'
    actor_entropy_scale: float = 1e-3
    actor_layers: int = 3
    actor_node_size: int = 100
    actor_train_noise: float = 0.4
    actor_eval_noise: float = 0.0
    actor_expl_min: float = 0.05
    actor_expl_decay: float = 7000.0
    actor_expl_type: str = 'epsilon_greedy'

    kl_loss_scale: float = 0.1
    discount_loss_scale: float = 5.0