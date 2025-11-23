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

    # Whether to use deterministic actions during evaluation.
    # For a random agent, this flag doesn’t really change behavior,
    # but we define it so RLTrainer can use a unified interface.
    eval_deterministic: bool = True
