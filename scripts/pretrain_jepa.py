# scripts/pretrain_jepa.py
# [JEPA] Pretrain JEPA encoder on collected ViZDoom frames (with EMA target).

from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm.auto import tqdm

from agents.jepa_model import JEPAConfig, JEPAModel
from dataset.jepa_frames_dataset import JEPAFramesDataset, JEPAFramesTemporalDataset

def build_dataset(frame_paths: Sequence[str], temporal_delta: int = 0):
    """
    temporal_delta == 0  -> single image JEPA (original)
    temporal_delta > 0   -> temporal JEPA, return (x_t, x_{t+delta})
    """
    if temporal_delta > 0:
        return JEPAFramesTemporalDataset(frame_paths, delta=temporal_delta)
    else:
        return JEPAFramesDataset(frame_paths)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain JEPA encoder on ViZDoom frames."
    )

    # --------------------------------------------------
    # Data paths
    # --------------------------------------------------
    parser.add_argument(
        "--frames_paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more .npy files with shape (N, C, H, W). "
            "Example: --frames_paths random.npy expert.npy"
        ),
    )
    parser.add_argument(
        "--frames_path",
        type=str,
        default=None,
        help="(Optional) Single .npy path, used if --frames_paths is not provided.",
    )

    # --------------------------------------------------
    # Model / training hyperparameters
    # --------------------------------------------------
    parser.add_argument(
        "--out_ckpt",
        type=str,
        required=True,
        help="Path to save JEPA checkpoint (.pt).",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=12,
        help="Number of input channels (e.g., frame_stack * 3).",
    )
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=256,
        help="Feature dimension of ConvEncoder.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine scheduler.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear warmup (0.0 means no warmup).",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument(
        "--temporal_delta",
        type=int,
        default=0,
        help="If >0, use temporal JEPA: predict frame t+delta from masked frame t.",
    )
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--var_weight", type=float, default=1.0)
    parser.add_argument("--covar_weight", type=float, default=7.0)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.99,
        help="EMA momentum for target encoder.",
    )
    parser.add_argument(
        "--std_target",
        type=float,
        default=1.0,
        help="Var Loss target, must larger thatn this value, or get penality.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    # --------------------------------------------------
    # Weights & Biases
    # --------------------------------------------------
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vizdoom-jepa",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity (team/user).",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Resolve frame paths
    # --------------------------------------------------
    if args.frames_paths is not None:
        frame_paths: List[str] = args.frames_paths
    elif args.frames_path is not None:
        frame_paths = [args.frames_path]
    else:
        raise ValueError(
            "You must provide either --frames_paths (one or more paths) "
            "or --frames_path (single path)."
        )

    print("[JEPA] Using frame files:")
    for p in frame_paths:
        print(f"  - {p}")

    # --------------------------------------------------
    # Dataset & DataLoader
    # --------------------------------------------------
    dataset = build_dataset(frame_paths, temporal_delta=args.temporal_delta)
    n_total, c, h, w = dataset.frames.shape
    if c != args.in_channels:
        raise ValueError(
            f"[JEPA] Channel mismatch: dataset has C={c}, "
            f"but --in_channels={args.in_channels}."
        )
    print(f"[JEPA] Total frames: {n_total}, frame shape: (C={c}, H={h}, W={w})")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------
    # Build JEPA model
    # --------------------------------------------------
    cfg = JEPAConfig(
        in_channels=args.in_channels,
        feat_dim=args.feat_dim,
        mask_ratio=args.mask_ratio,
        num_blocks=args.num_blocks,
        var_weight=args.var_weight,
        covar_weight=args.covar_weight,
        momentum=args.momentum,
        std_target=args.std_target,
    )
    model = JEPAModel(cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --------------------------------------------------
    # Cosine scheduler with warmup (step-based)
    # --------------------------------------------------
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(max(0.0, min(1.0, args.warmup_ratio)) * total_steps)

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=args.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        print(
            f"[JEPA] Total {total_steps} steps, steps_per_epoch = {steps_per_epoch}, epochs = {args.epochs}\n"
            f"[JEPA] Using Linear warmup for {warmup_steps} steps, \n"
            f"then CosineAnnealingLR for {total_steps - warmup_steps} steps.\n"
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.min_lr,
        )
        print(f"[JEPA] Using CosineAnnealingLR for {total_steps} steps (no warmup).")

    # --------------------------------------------------
    # Weights & Biases setup
    # --------------------------------------------------
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
        except ImportError:
            print("[JEPA] wandb is not installed; continuing without W&B logging.")
            wandb_run = None

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"[JEPA] Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            # temporal mode：batch is (x_t, x_tp)
            if isinstance(batch, (list, tuple)):
                x_ctx, x_target = (b.to(device, non_blocking=True) for b in batch)
                loss, stats = model(x_ctx, x_target)
            else:
                x = batch.to(device, non_blocking=True)
                loss, stats = model(x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # EMA update for target encoder
            model.update_target()

            epoch_loss += loss.item()
            global_step += 1

            # Current LR (all param groups share same LR)
            current_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(
                loss=f"{stats['loss_total'].item():.4f}",
                recon=f"{stats['loss_recon'].item():.4f}",
                var=f"{stats['loss_var'].item():.4f}",
                mask=f"{stats['mask_ratio_effective'].item():.2f}",
                lr=f"{current_lr:.2e}",
            )

            if wandb_run is not None:
                import wandb  # type: ignore

                wandb.log(
                    {
                        "train/loss_total": stats["loss_total"].item(),
                        "train/loss_recon": stats["loss_recon"].item(),
                        "train/loss_var": stats["loss_var"].item(),
                        "train/loss_covar": stats["loss_cov"].item(),
                        "train/cov_offdiag_abs_mean": stats["cov_offdiag_abs_mean"].item(),
                        "train/var_abs_mean": stats["var_abs_mean"].item(),
                        "train/mask_ratio_effective": stats[
                            "mask_ratio_effective"
                        ].item(),
                        "train/lr": current_lr,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"[JEPA] Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        if wandb_run is not None:
            import wandb  # type: ignore

            wandb.log(
                {"epoch/avg_loss": avg_loss, "epoch/index": epoch + 1},
                step=global_step,
            )

    # --------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------
    out_dir = os.path.dirname(args.out_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "jepa_state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
        },
        args.out_ckpt,
    )
    print(f"[JEPA] Saved JEPA checkpoint (encoder + model) to {args.out_ckpt}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
