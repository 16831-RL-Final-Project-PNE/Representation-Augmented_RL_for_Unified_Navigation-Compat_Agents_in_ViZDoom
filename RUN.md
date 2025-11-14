# Random Agent
# 1 run random agent training (Actually it is no training, just eval)
python -m scripts.train_random_basic

# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/random_basic_eval.npz \
  --out ./plots/random_basic_eval.png

# 3 output the playing game GIF using random agent
python -m scripts.eval_random_play \
  --scenario basic \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 320x240 \
  --gif ./out/basic_v2/best.gif \
  --gif_dir ./out/basic_v2/eps \
  --fps 8 \
  --gif_scale 1 \
  --gif_repeat 2 \
  --seed 0

# PPO Agent
# 1 run ppo agent training
python -m scripts.train_ppo_basic 

# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/basic_ppo_eval.npz \
  --out ./plots/basic_ppo_eval.png

# 3 output the playing game GIF using ppo agent
python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/ppo_basic_final.pt \
  --scenario basic \
  --episodes 5 \
  --gif ./out/ppo_basic/best.gif \
  --gif_dir ./out/ppo_basic/eps \
  --fps 8 \
  --gif_scale 1 \
  --gif_repeat 2 \
  --deterministic