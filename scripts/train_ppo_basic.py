# scripts/train_ppo_basic.py
import os

from env.doom_env import DoomEnv
from train.rl_trainer import RLTrainer
from configs.ppo_config import PPOConfig


def main():
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    train_env = DoomEnv(
        scenario="basic",
        frame_repeat=4,
        frame_stack=4,
        width=84,
        height=84,
        seed=0,
        window_visible=False,
        sound_enabled=False,
        base_res="320x240",
    )

    eval_env = DoomEnv(
        scenario="basic",
        frame_repeat=4,
        frame_stack=4,
        width=84,
        height=84,
        seed=42,
        window_visible=False,
        sound_enabled=False,
        base_res="320x240",
    )

    config = PPOConfig(
        total_iterations=50,
        steps_per_iteration=4096,
        batch_size=64,
        learning_rate=3e-4,
        eval_episodes=10,
        eval_interval=1,
        eval_log_path="./logs/basic_ppo_eval.npz",
        log_dir="./logs/tb_ppo_basic",
    )

    trainer = RLTrainer(
        env=train_env,
        eval_env=eval_env,
        agent_type="ppo",
        config=config,
        device=None,  # auto-select cuda if available
    )
    trainer.train()


if __name__ == "__main__":
    main()
