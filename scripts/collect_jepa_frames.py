# scripts/collect_jepa_frames.py
# [JEPA] Collect ViZDoom frames for JEPA pretraining, using a mix of
#        RandomAgent and a trained PPOAgent (if provided).

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
from tqdm.auto import trange, tqdm

import torch

from env.doom_env import DoomEnv
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
from pathlib import Path


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Mirror RLTrainer._obs_to_tensor:
      Input:  obs is (T, 3, H, W) uint8 from DoomEnv._get_obs()
      Output: (1, T*3, H, W) float32 in [0,1] on device.
    """
    obs_t = torch.from_numpy(obs)  # (T, 3, H, W)
    if obs_t.ndim != 4 or obs_t.shape[1] != 3:
        raise ValueError(f"[JEPA] Expected obs shape (T,3,H,W), got {obs_t.shape}")
    t, c, h, w = obs_t.shape
    obs_t = obs_t.reshape(1, t * c, h, w).float() / 255.0
    return obs_t.to(device)


def load_trained_ppo_agent(ckpt_path: str, device: torch.device) -> PPOAgent:
    """
    Load a PPOAgent from a checkpoint saved by RLTrainer.save_model().

    Expected checkpoint keys (from RLTrainer.save_model):
        {
            "agent_state_dict": self.agent.state_dict(),
            "agent_type": self.agent_type,
            "obs_shape": self.obs_shape,     # (frame_stack, 3, H, W)
            "n_actions": self.n_actions,
            "config": self.config,           # PPOConfig instance
        }
    """
    data = torch.load(ckpt_path, map_location=device, weights_only=False)

    agent_type = data.get("agent_type", "ppo")
    if agent_type != "ppo":
        raise ValueError(
            f"[JEPA] Checkpoint agent_type={agent_type}, expected 'ppo' "
            f"for PPOAgent."
        )

    obs_shape = data["obs_shape"]           # (frame_stack, 3, H, W)
    n_actions = int(data["n_actions"])
    config = data["config"]                 # PPOConfig instance

    # [JEPA] Match PPOAgent constructor exactly with training setup.
    feat_dim = getattr(config, "feat_dim", 256)
    backbone = getattr(config, "backbone", "cnn")
    freeze_backbone = getattr(config, "freeze_backbone", False)

    agent = PPOAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        feat_dim=feat_dim,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
    )
    agent.load_state_dict(data["agent_state_dict"])
    agent.to(device)
    agent.eval()

    return agent


# -------------------------------------------------------------
# [JEPA] Helper: select action from RandomAgent (expects torch.Tensor)
# -------------------------------------------------------------
def select_action_random(rand_agent: RandomAgent, obs: np.ndarray, device: torch.device) -> int:
    """
    Use RandomAgent to select an action.
    RandomAgent.act signature:
        actions, log_probs, values = act(obs_tensor)
    where obs_tensor is (B, C, H, W).
    """
    obs_t = _obs_to_tensor(obs, device=device)  # (1, C, H, W)
    with torch.no_grad():
        actions, log_probs, values = rand_agent.act(obs_t, deterministic=False)
    # actions: (B,) => take first element
    return int(actions[0].item())


def select_action_trained(
    agent: PPOAgent,
    obs: np.ndarray,
    device: torch.device,
    deterministic: bool = True,
) -> int:
    """
    Use trained PPOAgent to select an action.
    PPOAgent.act signature:
        actions, log_probs, values = act(obs_tensor, deterministic=...)
    """
    obs_t = _obs_to_tensor(obs, device=device)  # (1, C, H, W)
    with torch.no_grad():
        actions, log_probs, values = agent.act(obs_t, deterministic=deterministic)
    return int(actions[0].item())


def main() -> None:
    parser = argparse.ArgumentParser()

    # Environment & scenario (aligned with scripts/train_ppo_basic.py)
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
        help="Native ViZDoom render resolution before downsampling.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--num_steps",
        type=int,
        default=100_000,
        help="Total environment steps (frames) to collect.",
    )

    # ----- [JEPA] Trained agent & mixing options -----
    parser.add_argument(
        "--trained_ckpt",
        type=str,
        default=None,
        help=(
            "[JEPA] Path to PPO checkpoint saved by RLTrainer.save_model(). "
            "If None, collection uses only RandomAgent."
        ),
    )
    parser.add_argument(
        "--trained_rollout_prob",
        type=float,
        default=0.5,
        help=(
            "[JEPA] For each episode, probability of using the trained PPO "
            "policy (if provided). Otherwise use RandomAgent."
        ),
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="[JEPA] Max steps per episode before forcing reset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="[JEPA] Device for trained PPO agent (cuda or cpu).",
    )

    # ----- Output path -----
    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Path to save frames as .npy (N, C, H, W), C = frame_stack * 3.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ----- Build environment (same as training script) -----
    env = DoomEnv(
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

    # [JEPA] RandomAgent (always available)
    rand_agent = RandomAgent(n_actions=env.action_space_n)

    # [JEPA] Optional trained PPO agent
    ppo_agent: Optional[PPOAgent]
    if args.trained_ckpt is not None and args.trained_ckpt != "":
        print(f"[JEPA] Loading trained PPOAgent from {args.trained_ckpt}")
        ppo_agent = load_trained_ppo_agent(args.trained_ckpt, device=device)

        # Sanity check: env shape/action must match checkpoint
        # (We don't strictly need this, but it's good to fail fast.)
        ckpt_data = torch.load(args.trained_ckpt, map_location="cpu", weights_only=False)
        ckpt_obs_shape = tuple(ckpt_data["obs_shape"])
        ckpt_n_actions = int(ckpt_data["n_actions"])

        if tuple(env.observation_shape) != ckpt_obs_shape:
            raise ValueError(
                f"[JEPA] Env observation_shape={env.observation_shape}, "
                f"but checkpoint obs_shape={ckpt_obs_shape}. "
                f"Make sure frame_stack/width/height/base_res match training."
            )
        if env.action_space_n != ckpt_n_actions:
            raise ValueError(
                f"[JEPA] Env action_space_n={env.action_space_n}, "
                f"but checkpoint n_actions={ckpt_n_actions}. "
                f"Scenario/action_space mismatch?"
            )
    else:
        print("[JEPA] No trained_ckpt provided; collecting frames with RandomAgent only.")
        ppo_agent = None

    frames: List[np.ndarray] = []

    # Initial observation: shape (T, 3, H, W), T = frame_stack
    obs = env.reset()
    assert obs.ndim == 4 and obs.shape[1] == 3, f"Unexpected obs shape: {obs.shape}"

    # Choose controller for the first episode
    use_trained_episode = bool(
        ppo_agent is not None
        and np.random.rand() < args.trained_rollout_prob
    )
    ep_steps = 0
    steps_collected = 0

    with tqdm(total=args.num_steps, desc="[JEPA] Collecting frames") as pbar:
        while steps_collected < args.num_steps:
            # Save current stacked frames: (T,3,H,W) -> (C=T*3,H,W)
            T, C, H, W = obs.shape
            frame = obs.reshape(T * C, H, W)
            frames.append(frame.copy())

            steps_collected += 1
            pbar.update(1)

            # Select action
            if use_trained_episode and ppo_agent is not None:
                action = select_action_trained(
                    ppo_agent,
                    obs,
                    device=device,
                    deterministic=True,
                )
            else:
                action = select_action_random(
                    rand_agent,
                    obs,
                    device=device,
                )

            next_obs, reward, done, _info = env.step(action)
            ep_steps += 1

            # End episode if done or max_episode_steps reached
            if done or ep_steps >= args.max_episode_steps:
                obs = env.reset()
                ep_steps = 0
                use_trained_episode = bool(
                    ppo_agent is not None
                    and np.random.rand() < args.trained_rollout_prob
                )
            else:
                obs = next_obs

    frames_np = np.stack(frames, axis=0)  # (N, C, H, W)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, frames_np)
    print(f"[JEPA] Saved frames to {args.out_path}, shape={frames_np.shape}")


if __name__ == "__main__":
    main()
