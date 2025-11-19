# scripts/train_random_basic.py
import argparse
import os

from env.doom_env import DoomEnv
from train.rl_trainer import RLTrainer
from configs.random_config import RandomAgentConfig


def main():
    parser = argparse.ArgumentParser()

    # ----- Environment / training settings -----
    parser.add_argument(
        "--scenario",
        type=str,
        default="basic",
        choices=["basic", "my_way_home"],
        help="ViZDoom scenario to run.",
    )
    parser.add_argument("--total_iterations", type=int, default=20,
                        help="Number of high-level RL iterations (collect + eval).")
    parser.add_argument("--steps_per_iteration", type=int, default=4096,
                        help="Environment steps per iteration.")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes per evaluation call.")
    parser.add_argument("--frame_repeat", type=int, default=4,
                        help="Number of Doom tics to repeat each chosen action.")
    parser.add_argument("--frame_stack", type=int, default=4,
                        help="Number of consecutive frames to stack in observations.")
    parser.add_argument("--width", type=int, default=84,
                        help="Observation width after resizing.")
    parser.add_argument("--height", type=int, default=84,
                        help="Observation height after resizing.")
    parser.add_argument(
        "--base_res",
        type=str,
        default="320x240",
        choices=["160x120", "320x240", "800x600"],
        help="Native ViZDoom render resolution before downsampling.",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the training environment.")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Random seed for the eval environment.")

    # ----- Logging / plotting roots -----
    parser.add_argument(
        "--log_root",
        type=str,
        default="./logs",
        help="Root directory for eval npz logs and TensorBoard logs.",
    )
    parser.add_argument(
        "--plot_root",
        type=str,
        default="./plots",
        help="Root directory for plots (used by separate plotting scripts).",
    )

    # ----- TensorBoard + eval log file names -----
    parser.add_argument(
        "--tb_dirname",
        type=str,
        default=None,
        help="Subdirectory name under log_root for TensorBoard logs. "
             "Default: tb_random_<scenario>.",
    )
    parser.add_argument(
        "--eval_log_name",
        type=str,
        default=None,
        help="Eval npz file name under log_root. Default: random_<scenario>_eval.npz.",
    )

    # ----- Checkpoint settings -----
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="./checkpoints",
        help="Root directory to store model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Base name of the checkpoint file. Default: random_<scenario>.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0, RLTrainer can optionally save intermediate checkpoints every N iterations.",
    )

    args = parser.parse_args()

    # Ensure base directories exist
    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.plot_root, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Build environments
    # ------------------------------------------------------------------
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
        frame_repeat=args.frame_repeat,
        frame_stack=args.frame_stack,
        width=args.width,
        height=args.height,
        seed=args.eval_seed,
        window_visible=False,
        sound_enabled=False,
        base_res=args.base_res,
    )

    # ------------------------------------------------------------------
    # Build logging / checkpoint paths
    # ------------------------------------------------------------------
    scenario_tag = args.scenario

    # TensorBoard directory: <log_root>/<tb_dirname>
    tb_dirname = args.tb_dirname or f"tb_random_{scenario_tag}"
    tb_log_dir = os.path.join(args.log_root, tb_dirname)

    # Eval npz path: <log_root>/<eval_log_name>
    eval_log_name = args.eval_log_name or f"random_{scenario_tag}_eval.npz"
    eval_log_path = os.path.join(args.log_root, eval_log_name)

    # Checkpoints
    checkpoint_dir = args.checkpoint_root
    checkpoint_name = args.checkpoint_name or f"random_{scenario_tag}"

    # RandomAgentConfig is tailored for the random baseline, but has all fields
    # needed by RLTrainer.
    config = RandomAgentConfig(
        total_iterations=args.total_iterations,
        steps_per_iteration=args.steps_per_iteration,
        eval_episodes=args.eval_episodes,
        eval_interval=1,
        eval_log_path=eval_log_path,
        log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        save_every=args.save_every,
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
