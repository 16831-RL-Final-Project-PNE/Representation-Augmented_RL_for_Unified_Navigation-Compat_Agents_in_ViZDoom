# scripts/train_ppo_basic.py
import argparse
import os

from env.doom_env import DoomEnv
from train.rl_trainer import RLTrainer
from configs.ppo_config import PPOConfig


def main():
    parser = argparse.ArgumentParser()

    # ----- Environment & scenario -----
    parser.add_argument(
        "--scenario",
        type=str,
        default="basic",
        choices=["basic", "my_way_home"],
        help="ViZDoom scenario to run.",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="no_shoot",
        choices=["usual", "no_shoot"],
        help="ViZDoom action space to run.",
    )
    parser.add_argument("--frame_repeat", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument(
        "--base_res",
        type=str,
        default="320x240",
        choices=["160x120", "320x240", "800x600"],
        help="Native ViZDoom render resolution.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=42)

    # ----- PPO training hyper-parameters -----
    parser.add_argument("--total_iterations", type=int, default=200)
    parser.add_argument("--steps_per_iteration", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.25)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # ----- Evaluation settings -----
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument(
        "--eval_log_dir",
        type=str,
        default="./logs",
        help="Directory to store eval npz logs.",
    )
    parser.add_argument(
        "--eval_log_name",
        type=str,
        default="basic_ppo_eval.npz",
        help="Filename for eval npz log.",
    )
    parser.add_argument(
        "--tb_log_dir",
        type=str,
        default="./logs/tb_basic_ppo",
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--eval_deterministic",
        action="store_true",
        help="Use deterministic policy during eval if set.",
    )

    # ----- Checkpoint settings -----
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save PPO checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="basic_ppo",
        help="Checkpoint file prefix (e.g., basic_ppo_final.pt).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="0 = only final checkpoint; >0 = also every N iterations.",
    )

    parser.add_argument(
        "--feat_dim",
        type=int,
        default=256,
        help="feature dimension output from encoder part.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="cnn",
        choices=["cnn", "dinov2", "dinov3"],
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the image encoder.",
    )

    # ----- RND options -----
    parser.add_argument(
        "--use_rnd",
        action="store_true",
        help="Enable RND intrinsic reward for exploration.",
    )
    parser.add_argument("--rnd_int_coef", type=float, default=1.0,
                        help="Coefficient on intrinsic (RND) reward.")
    parser.add_argument("--rnd_ext_coef", type=float, default=1.0,
                        help="Coefficient on extrinsic reward when mixing.")
    parser.add_argument("--rnd_gamma", type=float, default=0.99,
                        help="EMA factor for intrinsic reward std.")
    parser.add_argument("--rnd_lr", type=float, default=1e-4,
                        help="Learning rate for RND predictor.")
    parser.add_argument("--rnd_weight_decay", type=float, default=1e-4,
                        help="Weight decay for RND predictor AdamW.")
    parser.add_argument("--rnd_batch_size", type=int, default=256,
                        help="RND predictor batch size per rollout.")
    parser.add_argument("--rnd_epochs", type=int, default=1,
                        help="Number of passes over RND data per rollout.")

    args = parser.parse_args()

    # ----- Create needed directories -----
    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    eval_log_path = os.path.join(args.eval_log_dir, args.eval_log_name)

    # ----- Build train & eval environments -----
    train_env = DoomEnv(
        scenario=args.scenario,
        frame_repeat=args.frame_repeat,
        frame_stack=args.frame_stack,
        width=args.width,
        height=args.height,
        seed=args.seed,
        window_visible=False,
        sound_enabled=False,
        base_res=args.base_res,
    )

    eval_env = DoomEnv(
        scenario=args.scenario,
        action_space=args.action_space,
        frame_repeat=args.frame_repeat,
        frame_stack=args.frame_stack,
        width=args.width,
        height=args.height,
        seed=args.eval_seed,
        window_visible=False,
        sound_enabled=False,
        base_res=args.base_res,
    )

    # ----- PPO config -----
    config = PPOConfig(
        total_iterations=args.total_iterations,
        steps_per_iteration=args.steps_per_iteration,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        eval_log_path=eval_log_path,
        log_dir=args.tb_log_dir,
        eval_deterministic=args.eval_deterministic,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        save_every=args.save_every,
        backbone=args.backbone,
        feat_dim=args.feat_dim,
        freeze_backbone=args.freeze_backbone,
        use_rnd=args.use_rnd,
        rnd_int_coef=args.rnd_int_coef,
        rnd_ext_coef=args.rnd_ext_coef,
        rnd_gamma=args.rnd_gamma,
        rnd_lr=args.rnd_lr,
        rnd_weight_decay=args.rnd_weight_decay,
        rnd_batch_size=args.rnd_batch_size,
        rnd_epochs=args.rnd_epochs,
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
