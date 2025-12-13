# eval/plot_tb_avg_return.py
#
# Read multiple TensorBoard logdirs, extract a scalar (default: Eval_AverageReturn),
# and plot value vs. step for all runs on a single figure.
#
# Example:
#   python -m eval.plot_tb_avg_return \
#       --logdirs /data/.../tb_run1 /data/.../tb_run2 \
#       --tag Eval_AverageReturn \
#       --output vizdoom_avg_return.png

import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar_from_logdir(logdir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (steps, values) for a given scalar tag from a TensorBoard logdir."""
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f"Logdir does not exist or is not a directory: {logdir}")

    # size_guidance["scalars"]=0 means "load all scalar events"
    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()

    scalar_tags = acc.Tags().get("scalars", [])
    if tag not in scalar_tags:
        raise ValueError(
            f"Tag '{tag}' not found in logdir '{logdir}'. "
            f"Available scalar tags: {scalar_tags}"
        )

    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float32)

    # Sort by step in case events are out of order
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]

    return steps, values


def plot_runs(
    logdirs: List[str],
    tag: str,
    output_path: str,
    title: str = "Average Return vs. Training Iterations",
) -> None:
    """Plot tag vs. step for multiple TensorBoard runs and save as PNG."""
    plt.figure(figsize=(8, 5))

    for logdir in logdirs:
        try:
            steps, values = load_scalar_from_logdir(logdir, tag)
        except Exception as e:
            print(f"[WARN] Skipping logdir '{logdir}': {e}")
            continue

        # Legend name = last folder name of the logdir
        label = os.path.basename(os.path.normpath(logdir))
        plt.plot(steps, values, label=label)

    if not plt.gca().has_data():
        raise RuntimeError("No valid runs were plotted; check logdirs and tag name.")

    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"[INFO] Plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a scalar (e.g., Eval_AverageReturn) vs. step from multiple "
            "TensorBoard logdirs into a single PNG."
        )
    )
    parser.add_argument(
        "--logdirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more TensorBoard log directories.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Eval_AverageReturn",
        help="Scalar tag name to plot (default: Eval_AverageReturn).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="avg_return_comparison.png",
        help="Output PNG path (default: avg_return_comparison.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Average Return vs. Training Iterations",
        help="Plot title.",
    )

    args = parser.parse_args()
    plot_runs(args.logdirs, args.tag, args.output, title=args.title)


if __name__ == "__main__":
    main()
