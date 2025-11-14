# eval/evaluation.py
import argparse
from typing import List, Tuple

import numpy as np
import torch


def stacked_obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert stacked frames (T, 3, H, W) from DoomEnv into a (1, C, H, W)
    float32 tensor in [0, 1] on the given device.
    """
    obs_t = torch.from_numpy(obs)  # (T, 3, H, W)
    if obs_t.ndim != 4:
        raise ValueError(f"Expected obs with 4 dims (T,3,H,W), got {obs_t.shape}")

    t, c, h, w = obs_t.shape

    # Use reshape instead of view to handle non-contiguous tensors safely.
    obs_t = obs_t.reshape(1, t * c, h, w).float() / 255.0
    return obs_t.to(device)

@torch.no_grad()
def evaluate_policy(
    env,
    agent,
    num_episodes: int,
    device: torch.device,
    deterministic: bool = True,
    return_raw: bool = False,
):
    """
    Run evaluation episodes.

    If return_raw=False (default): return (mean_return, std_return).
    If return_raw=True:  return (returns_list, ep_len_list).
    """
    returns: List[float] = []
    ep_lens: List[int] = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            obs_tensor = stacked_obs_to_tensor(obs, device)
            action, _, _ = agent.act(obs_tensor, deterministic=deterministic)
            obs, reward, done, _info = env.step(int(action.item()))
            ep_ret += float(reward)
            ep_len += 1
        returns.append(ep_ret)
        ep_lens.append(ep_len)

    if return_raw:
        return returns, ep_lens

    returns_np = np.asarray(returns, dtype=np.float32)
    return float(returns_np.mean()), float(returns_np.std())


class EvalLogger:
    """
    Keeps track of evaluation statistics (mean return per iteration)
    and can save/load from disk in NumPy format.
    """
    def __init__(self) -> None:
        self.iterations: List[int] = []
        self.mean_returns: List[float] = []
        self.std_returns: List[float] = []

    def add(self, iteration: int, mean_return: float, std_return: float) -> None:
        self.iterations.append(int(iteration))
        self.mean_returns.append(float(mean_return))
        self.std_returns.append(float(std_return))

    def save(self, path: str) -> None:
        np.savez(
            path,
            iterations=np.asarray(self.iterations, dtype=np.int32),
            mean_returns=np.asarray(self.mean_returns, dtype=np.float32),
            std_returns=np.asarray(self.std_returns, dtype=np.float32),
        )

    @staticmethod
    def load(path: str) -> "EvalLogger":
        data = np.load(path)
        logger = EvalLogger()
        logger.iterations = data["iterations"].tolist()
        logger.mean_returns = data["mean_returns"].tolist()
        logger.std_returns = data["std_returns"].tolist()
        return logger


def plot_eval_curve(npz_path: str, out_path: str, annotate: bool = False, annotate_last_only: bool = False) -> None:
    """
    Plot average eval return vs iteration, homework-style.
    """
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    iters = data["iterations"]
    mean = data["mean_returns"]
    std = data["std_returns"]

    plt.figure()
    plt.plot(iters, mean, marker="o")
    plt.fill_between(iters, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Average Eval Return")
    plt.title("Average Evaluation Return vs Iteration")

    # Label values near each point
    if annotate_last_only:
        plt.text(iters[-1], mean[-1], f"{mean[-1]:.1f}", fontsize=8, ha="center", va="bottom")
    elif annotate:
        for x, y in zip(iters, mean):
            plt.text(x, y, f"{y:.1f}", fontsize=8, ha="center", va="bottom")    

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved eval curve to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True, help="Path to .npz eval log")
    parser.add_argument("--out", type=str, default="./plots/eval_return.png")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each point with its mean return value.",
    )
    parser.add_argument(
        "--annotate_last_only",
        action="store_true",
        help="Annotate only the last point with its mean return value.",
    )
    args = parser.parse_args()
    plot_eval_curve(args.log_path, args.out, args.annotate, args.annotate_last_only)
