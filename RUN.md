# Random Agent
# 1 run random agent training (Actually it is no training, just eval)
python -m scripts.train_random_basic --scenario basic --action_space usual

python -m scripts.train_random_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 20 \
  --steps_per_iteration 4096 \
  --log_root ./logs_mwh \
  --plot_root ./plots_mwh \
  --tb_dirname run1_tb \
  --eval_log_name random_mwh_run1_eval.npz \
  --checkpoint_root ./checkpoints_mwh \
  --checkpoint_name random_mwh_run1 \
  --save_every 0

# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/random_basic_eval.npz \
  --out ./plots/random_basic_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs_mwh/random_mwh_run1_eval.npz \
  --out ./plots_mwh/random_mwh_run1_eval.png \
  --annotate_last_only

# 3 output the playing game GIF using random agent
python -m scripts.eval_random_play \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/random_basic_v7/best.gif \
  --gif_dir ./out/random_basic_v7/eps \
  --fps 12 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0

python -m scripts.eval_random_play \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 320x240 \
  --gif ./out/random_mwh_v2/best.gif \
  --gif_dir ./out/random_mwh_v2/eps \
  --fps 12 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0

# PPO Agent
# 1 run ppo agent training
python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir ./logs \
  --eval_log_name basic_ppo_eval.npz \
  --tb_log_dir ./logs/tb_ppo_basic \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name ppo_basic \
  --save_every 0

python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 400 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir ./logs \
  --eval_log_name mwh_ppo_eval_v2.npz \
  --tb_log_dir ./logs/tb_ppo_mwh_v2 \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name ppo_mwh_v2 \
  --save_every 0

# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/basic_ppo_eval.npz \
  --out ./plots/basic_ppo_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs/mwh_ppo_eval.npz \
  --out ./plots/mwh_ppo_eval.png \
  --annotate_last_only

# 3 output the playing game GIF using ppo agent
python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/ppo_basic_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 320x240 \
  --gif ./out/ppo_basic_v4/best.gif \
  --gif_dir ./out/ppo_basic_v4/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/ppo_basic_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/ppo_basic_v5/best.gif \
  --gif_dir ./out/ppo_basic_v5/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/ppo_mwh_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/ppo_mwh_v1/best.gif \
  --gif_dir ./out/ppo_mwh_v1/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic