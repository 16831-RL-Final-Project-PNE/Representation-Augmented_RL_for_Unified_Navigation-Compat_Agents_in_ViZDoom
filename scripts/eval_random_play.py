# scripts/run_random_agent.py
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import trange

from env.doom_env import DoomEnv
from agents.random_agent import RandomAgent


def _maybe_upscale(frame_hwc: np.ndarray, scale: int | None) -> np.ndarray:
    """
    Optionally upscale an HWC uint8 frame by an integer factor using nearest-neighbor.
    """
    if not scale or scale == 1:
        return frame_hwc
    from PIL import Image

    h, w, _ = frame_hwc.shape
    img = Image.fromarray(frame_hwc)
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def run_random(
    env: DoomEnv,
    agent: RandomAgent,
    episodes: int = 10,
    gif_path: str | None = None,
    gif_dir: str | None = None,
    fps: int = 8,
    step_frame_repeat_for_gif: int = 1,
    gif_scale: int = 2,
) -> None:
    """
    Run a random policy (via RandomAgent) for a number of episodes.

    step_frame_repeat_for_gif: how many times to write each decision frame to the GIF timeline.
      Use >1 to make very short episodes look animated even with frameskip.
    gif_scale: integer upscale factor for nicer viewing.
    """
    best_return, best_frames = -1e9, []
    total_return = 0.0

    for i in trange(episodes, desc="Random episodes"):
        obs = env.reset()
        ep_frames: list[np.ndarray] = []
        ep_ret = 0.0
        ep_tics = 0
        last_info: dict = {}

        # Record the initial frame for the episode
        f0 = env.render("rgb_array")
        if f0 is not None:
            fr = _maybe_upscale(f0, gif_scale)
            ep_frames.append(fr)

        done = False
        while not done:
            # RandomAgent ignores the content of obs; it only needs batch size.
            dummy_obs = torch.zeros(1, 1)  # shape (B, anything)
            actions, _, _ = agent.act(dummy_obs)
            a = int(actions.item())

            obs, r, done, info = env.step(a)
            ep_ret += float(r)
            ep_tics += env.frame_repeat
            last_info = info

            frames = info.get("tic_frames")
            if frames:
                for frm in frames:
                    fr = _maybe_upscale(frm, gif_scale)
                    for _ in range(max(1, step_frame_repeat_for_gif)):
                        ep_frames.append(fr)
            else:
                # Fallback: use the latest rendered frame
                f = env.render("rgb_array")
                if f is not None:
                    fr = _maybe_upscale(f, gif_scale)
                    for _ in range(max(1, step_frame_repeat_for_gif)):
                        ep_frames.append(fr)

        # Prefer the terminal step's info (has reason/summary); fall back to env.episode_summary()
        ep_info = last_info.get("episode", env.episode_summary())
        reason = ep_info.get("reason", None)
        kills = ep_info.get("kills", last_info.get("kills", 0))
        secs = ep_info.get("secs", ep_tics / 35.0)

        total_return += ep_ret
        print(
            f"Episode {i + 1}/{episodes}: return={ep_ret:.2f}, "
            f"KILLCOUNT={kills}, time={secs:.1f}s ({ep_tics} tics)"
            + (f", reason={reason}" if reason else "")
        )

        if gif_dir and ep_frames:
            os.makedirs(gif_dir, exist_ok=True)
            out_file = os.path.join(
                gif_dir,
                f"ep_{i + 1:03d}_return_{ep_ret:.2f}.gif",
            )
            imageio.mimsave(out_file, ep_frames, duration=1.0 / max(1, fps), loop=0)

        if ep_ret > best_return and ep_frames:
            best_return, best_frames = ep_ret, ep_frames

    avg_return = total_return / max(1, episodes)
    print(
        f"\nEpisodes: {episodes} | Average return: {avg_return:.2f} | "
        f"Best: {best_return:.2f}"
    )

    if gif_path and best_frames:
        out_dir = os.path.dirname(gif_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        root, ext = os.path.splitext(os.path.basename(gif_path))
        out_file = os.path.join(
            out_dir,
            f"{root}_return_{best_return:.2f}{ext}",
        )
        imageio.mimsave(out_file, best_frames, duration=1.0 / max(1, fps), loop=0)
        print(f"Saved best-episode GIF to: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["basic", "my_way_home"],
        default="basic",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frame_repeat", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", type=str, default="./out/best.gif")
    parser.add_argument("--gif_dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--gif_scale",
        type=int,
        default=1,
        help="integer upscale for saved GIFs",
    )
    parser.add_argument(
        "--gif_repeat",
        type=int,
        default=2,
        help="repeat each decision frame in GIF",
    )
    parser.add_argument(
        "--base_res",
        type=str,
        default="320x240",
        choices=["160x120", "320x240", "800x600"],
        help="native ViZDoom render resolution",
    )
    args = parser.parse_args()

    env = DoomEnv(
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

    # RandomAgent baseline
    agent = RandomAgent(n_actions=env.action_space_n)

    run_random(
        env=env,
        agent=agent,
        episodes=args.episodes,
        gif_path=args.gif,
        gif_dir=args.gif_dir,
        fps=args.fps,
        step_frame_repeat_for_gif=args.gif_repeat,
        gif_scale=args.gif_scale,
    )
    env.close()


if __name__ == "__main__":
    main()
