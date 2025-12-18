#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/patrick/hf_cache

# Optional: pick a GPU
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

python -m eval.plot_tb_avg_return \
  --logdirs \
    /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_frozen \
    /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_unfreeze1 \
    /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_unfreeze2 \
    /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_fullft \
  --tag Eval_AverageReturn \
  --output ./plots/mwh_eval_td3frozen_trans_return.png