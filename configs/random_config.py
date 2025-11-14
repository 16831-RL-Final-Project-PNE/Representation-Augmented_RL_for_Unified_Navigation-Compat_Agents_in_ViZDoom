# configs/random_config.py
from dataclasses import dataclass


@dataclass
class RandomAgentConfig:
    """
    Configuration for evaluating a random agent over multiple iterations.

    Note:
      Fields are chosen to be compatible with RLTrainer, even if some of them
      are not used by the random agent (e.g., PPO-specific hyperparameters).
    """

    # High-level training / evaluation schedule
    total_iterations: int = 20
    steps_per_iteration: int = 4096
    eval_episodes: int = 10
    eval_interval: int = 1
    eval_log_path: str = "./logs/random_basic_eval.npz"

    # Logging / checkpointing
    log_dir: str = "./logs/tb_random_basic"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "random_basic"
    save_every: int = 0  # 0 = only save final

    # The following fields are mostly unused by the random agent, but kept
    # for compatibility with RLTrainer and RolloutBuffer.
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO-specific fields (not used when agent_type == "random", but required
    # so that RLTrainer can access these attributes safely)
    ppo_epochs: int = 1
    clip_coef: float = 0.0
    value_coef: float = 0.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.0
