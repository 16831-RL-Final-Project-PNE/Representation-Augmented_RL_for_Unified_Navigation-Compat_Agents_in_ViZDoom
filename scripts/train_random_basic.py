# scripts/train_random_basic.py
import os

from env.doom_env import DoomEnv
from train.rl_trainer import RLTrainer
from configs.random_config import RandomAgentConfig


def main():
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

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

    # RandomAgentConfig is tailored for the random baseline, but has all fields
    # needed by RLTrainer.
    config = RandomAgentConfig(
        total_iterations=20,
        steps_per_iteration=4096,
        eval_episodes=10,
        eval_interval=1,
        eval_log_path="./logs/random_basic_eval.npz",
        log_dir="./logs/tb_random_basic",
        checkpoint_dir="./checkpoints",
        checkpoint_name="random_basic",
        save_every=0,
    )

    trainer = RLTrainer(
        env=train_env,
        eval_env=eval_env,
        agent_type="random",   # use the random agent path
        config=config,
        device=None,           # auto-select cuda if available
    )
    trainer.train()


if __name__ == "__main__":
    main()
