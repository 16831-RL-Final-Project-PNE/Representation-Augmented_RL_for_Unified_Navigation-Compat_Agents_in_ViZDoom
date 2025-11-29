#!/usr/bin/env python
"""
Evaluate a trained PPO policy from a checkpoint with deterministic actions.

Usage example:

python -m scripts.eval_ppo_checkpoint \
  --scenario basic \
  --action_space usual \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_iter199.pt \
  --eval_log_path ./logs/basic_ppo_eval_det.npz \
  --episodes 50 \
  --backbone cnn \
  --feat_dim 256

This does NOT train anything. It:
  1) builds the DoomEnv,
  2) builds a PPOAgent with the requested backbone,
  3) loads the checkpoint weights,
  4) runs N episodes with agent.act(..., deterministic=True),
  5) saves per-episode returns to a .npz file.
"""

import argparse
import os
from typing import Any, Dict

import numpy as np
import torch

from env.doom_env import DoomEnv
from agents.ppo_agent import PPOAgent
from eval.evaluation import EvalLogger, stacked_obs_to_tensor


def make_env(args: argparse.Namespace) -> DoomEnv:
    """Create a single DoomEnv for evaluation."""
    env = DoomEnv(
        scenario=args.scenario,
        action_space=args.action_space,
        frame_repeat=args.frame_repeat,
        frame_stack=args.frame_stack,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    return env


def infer_obs_shape(env: DoomEnv) -> tuple[int, int, int, int]:
    """
    PPOAgent expects obs_shape = (frame_stack, 3, H, W).
    DoomEnv 可能有 observation_shape 或 observation_space.shape。
    """
    if hasattr(env, "observation_shape"):
        # already (frame_stack, 3, H, W)
        return tuple(int(x) for x in env.observation_shape)

    # fallback: assume (C,H,W) and C = 3 * frame_stack
    c, h, w = env.observation_space.shape  # type: ignore[attr-defined]
    assert c % 3 == 0, f"Expected C divisible by 3, got {c}"
    frame_stack = c // 3
    return (frame_stack, 3, h, w)


def get_n_actions(env: DoomEnv) -> int:
    """Get number of discrete actions from env."""
    if hasattr(env, "action_space_n"):
        return int(env.action_space_n)  # type: ignore[attr-defined]
    if hasattr(env, "action_space"):
        return int(env.action_space)
    if hasattr(env, "n_actions"):
        return int(env.n_actions)
    raise RuntimeError("Cannot infer number of actions from env.")


def load_agent_from_checkpoint(
    ckpt_path: str,
    env: DoomEnv,
    backbone: str,
    feat_dim: int,
    freeze_backbone: bool,
    device: torch.device,
) -> PPOAgent:
    """Build PPOAgent and load weights from checkpoint."""
    obs_shape = infer_obs_shape(env)
    n_actions = get_n_actions(env)

    agent = PPOAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        feat_dim=feat_dim,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
    ).to(device)

    # PyTorch 2.6: default weights_only=True will fail because ckpt contains PPOConfig.
    # This checkpoint is from our own code, so it's safe to disable weights_only.
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt: Dict[str, Any] = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,  # <--- allow to load self-defined class ppo_config.py
    )

    # Heuristics: support several common formats
    if isinstance(ckpt, dict):
        if "agent_state_dict" in ckpt:
            state_dict = ckpt["agent_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # assume the dict itself is a state_dict
            state_dict = ckpt
    else:
        # rare case: checkpoint is already a raw state_dict
        state_dict = ckpt

    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading state_dict: {unexpected}")

    agent.eval()
    return agent


def run_eval(
    env: DoomEnv,
    agent: PPOAgent,
    episodes: int,
    device: torch.device,
    deterministic: bool = True,
) -> np.ndarray:
    """Run multiple evaluation episodes and return per-episode returns."""
    returns: list[float] = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # obs: (C,H,W) numpy -> (1,C,H,W) tensor
            # obs: (T,3,H,W) numpy -> (1, 3T, H, W) tensor (same as training)
            obs_t = stacked_obs_to_tensor(obs, device)
            
            with torch.no_grad():
                actions, logp, value = agent.act(obs_t, deterministic=deterministic)
            action = int(actions[0].item())

            step_out = env.step(action)
            # support Gym-like two variants: (obs, reward, done, info)
            # or (obs, reward, terminated, truncated, info)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            elif len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                raise RuntimeError(f"Unexpected env.step() output: len={len(step_out)}")

            ep_return += float(reward)

        returns.append(ep_return)
        print(f"[INFO] Episode {ep+1}/{episodes}: return = {ep_return:.2f}")

    return np.asarray(returns, dtype=np.float32)

def save_eval_log(
    returns: np.ndarray,
    out_path: str,
    extra_meta: Dict[str, Any],
) -> None:
    """
    Save evaluation result in the SAME format as EvalLogger.save(),
    so that eval/evaluation.py:plot_eval_curve can read it directly.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    print(f"[INFO] Mean return over {len(returns)} episodes: {mean_ret:.2f} ± {std_ret:.2f}")

    logger = EvalLogger()
    # only evaluate one checkpoint, take it as 'iteration 0'
    logger.add(iteration=0, mean_return=mean_ret, std_return=std_ret)
    logger.save(out_path)

    # extra metadata, you can save another json or npz if you want;
    # for compatibility with evaluation.py, not merging into same file.
    meta_path = out_path.replace(".npz", "_meta.npz")
    np.savez(meta_path, **extra_meta)
    print(f"[INFO] Saved eval curve log to {out_path}")
    print(f"[INFO] Saved extra metadata to {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic evaluation of a PPO checkpoint on DoomEnv")

    # Env config (match your train_ppo_basic defaults)
    parser.add_argument("--scenario", type=str, default="basic", help="ViZDoom scenario, e.g. basic / my_way_home")
    parser.add_argument("--action_space", type=str, default="usual", help="Action space name, e.g. usual / no_shoot")
    parser.add_argument("--frame_repeat", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--seed", type=int, default=0)

    # Agent / backbone
    parser.add_argument("--backbone", type=str, default="cnn", choices=["cnn", "dinov2", "dinov3"])
    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze DINO backbone during forward (match training config); ignored for cnn.",
    )

    # Checkpoint / eval
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument(
        "--eval_log_path",
        type=str,
        default="./logs/ppo_eval_det.npz",
        help="Where to save .npz with per-episode returns",
    )
    parser.add_argument(
        "--stochastic_eval",
        action="store_true",
        help="If set, sample actions instead of deterministic argmax (default is deterministic).",
    )

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build env & agent
    env = make_env(args)
    agent = load_agent_from_checkpoint(
        ckpt_path=args.checkpoint,
        env=env,
        backbone=args.backbone,
        feat_dim=args.feat_dim,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )

    deterministic = not args.stochastic_eval
    print(f"[INFO] Evaluating with deterministic={deterministic}")

    # Run eval
    returns = run_eval(
        env=env,
        agent=agent,
        episodes=args.episodes,
        device=device,
        deterministic=deterministic,
    )

    # Save .npz
    meta = dict(
        scenario=args.scenario,
        action_space=args.action_space,
        backbone=args.backbone,
        feat_dim=args.feat_dim,
        deterministic=deterministic,
        checkpoint=os.path.abspath(args.checkpoint),
    )
    save_eval_log(returns, args.eval_log_path, extra_meta=meta)

    env.close()


if __name__ == "__main__":
    main()
