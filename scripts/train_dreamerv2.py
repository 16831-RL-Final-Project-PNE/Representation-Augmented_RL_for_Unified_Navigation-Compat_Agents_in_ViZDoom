# scripts/train_ppo_basic.py
import argparse
import os

from env.doom_env import DoomEnv
from train.rl_trainer import RLTrainer
from configs.dreamerv2_config import DreamerV2Config


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
    parser.add_argument("--frame_repeat", type=int, default=1)
    parser.add_argument("--frame_stack", type=int, default=1)
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

    # ----- DreamerV2 training hyper-parameters -----
    # TODO: Update these hyperparameters
    parser.add_argument("--total_iterations", type=int, default=200)
    parser.add_argument("--steps_per_iteration", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=128)

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
        default="basic_dreamerv2_eval.npz",
        help="Filename for eval npz log.",
    )
    parser.add_argument(
        "--tb_log_dir",
        type=str,
        default="./logs/tb_dreamerv2_basic",
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
        help="Directory to save DreamerV2 checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="dreamerv2_basic",
        help="Checkpoint file prefix (e.g., dreamerv2_basic_final.pt).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="0 = only final checkpoint; >0 = also every N iterations.",
    )

    parser.add_argument(
        "--use_dino_v3",
        action="store_true",
        help="Use DinoV3 encoder if set.",
    )

    args = parser.parse_args()

    # ----- Create needed directories -----
    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    eval_log_path = os.path.join(args.eval_log_dir, args.eval_log_name)

    # ----- Build train & eval environments -----
    train_env = DoomEnv(
        scenario=args.scenario,
        action_space=args.action_space,
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

    # ----- DreamerV2 config -----
    config = DreamerV2Config(
        total_iterations=args.total_iterations,
        steps_per_iteration=args.steps_per_iteration,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        eval_log_path=eval_log_path,
        log_dir=args.tb_log_dir,
        eval_deterministic=args.eval_deterministic,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        save_every=args.save_every,
        use_dino_v3=args.use_dino_v3,
    )

    trainer = RLTrainer(
        env=train_env,
        eval_env=eval_env,
        agent_type="dreamerv2",
        config=config,
        device=None,  # auto-select cuda if available
    )
    trainer.train()


if __name__ == "__main__":
    main()
