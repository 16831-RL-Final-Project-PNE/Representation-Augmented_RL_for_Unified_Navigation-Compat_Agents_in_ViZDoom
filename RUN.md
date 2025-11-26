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
  --tb_dirname tb_run1 \
  --eval_log_name mwh_random_run1_eval.npz \
  --checkpoint_root ./checkpoints_mwh \
  --checkpoint_name mwh_random_run1 \
  --save_every 0

# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/basic_random_eval.npz \
  --out ./plots/basic_random_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs_mwh/mwh_random_run1_eval.npz \
  --out ./plots_mwh/mwh_random_run1_eval.png \
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
  --gif ./out/basic_random_v7/best.gif \
  --gif_dir ./out/basic_random_v7/eps \
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
  --gif ./out/mwh_random_v2/best.gif \
  --gif_dir ./out/mwh_random_v2/eps \
  --fps 12 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0

# PPO Agent
# 1 run ppo agent training
CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
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
  --tb_log_dir ./logs/tb_basic_ppo \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name basic_ppo \
  --save_every 40

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 2.5e-4 \
  --clip_coef 0.1 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.3 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --eval_log_dir ./logs \
  --eval_log_name basic_ppo_rnd_eval.npz \
  --tb_log_dir ./logs/tb_basic_ppo_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_rnd \
  --save_every 80

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
  --tb_log_dir ./logs/tb_mwh_ppo_v2 \
  --checkpoint_dir ./checkpoints_mwh \
  --checkpoint_name mwh_ppo_v2 \
  --save_every 0

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone cnn \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.5 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --eval_log_dir ./logs \
  --eval_log_name mwh_ppo_rnd_eval.npz \
  --tb_log_dir ./logs/tb_mwh_ppo_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name mwh_ppo_rnd \
  --save_every 80

# need modified filename
CUDA_VISIBLE_DEVICES=7 python -m scripts.train_ppo_basic \
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
  --eval_log_name basic_ppo_dinov3_eval.npz \
  --tb_log_dir ./logs/tb_basic_ppo_dinov3 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov3 \
  --save_every 100 \
  --backbone dinov3 \
  --freeze_backbone

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --backbone dinov3 \
  --freeze_backbone \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.2 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --eval_log_dir ./logs \
  --eval_log_name basic_ppo_dinov3_rnd_eval.npz \
  --tb_log_dir ./logs/tb_basic_ppo_dinov3_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov3_rnd \
  --save_every 100

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
  --eval_log_name mwh_ppo_dinov3_eval_v3.npz \
  --tb_log_dir ./logs/tb_ppo_dinov3_mwh_v3 \
  --checkpoint_dir ./checkpoints_mwh \
  --checkpoint_name ppo_dinov3_mwh_v3 \
  --save_every 80 \
  --backbone dinov3 \
  --freeze_backbone

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone dinov3 \
  --freeze_backbone \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.4 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --eval_log_dir ./logs \
  --eval_log_name mwh_ppo_dinov3_rnd_eval.npz \
  --tb_log_dir ./logs/tb_mwh_ppo_dinov3_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name mwh_ppo_dinov3_rnd \
  --save_every 80

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
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
  --eval_log_name basic_ppo_dinov2_eval.npz \
  --tb_log_dir ./logs/tb_basic_ppo_dinov2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov2 \
  --save_every 100 \
  --backbone dinov2 \
  --freeze_backbone

CUDA_VISIBLE_DEVICES=1 python -m scripts.train_ppo_basic \
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
  --eval_log_name mwh_ppo_dinov2_eval.npz \
  --tb_log_dir ./logs/tb_mwh_ppo_dinov2 \
  --checkpoint_dir ./checkpoints_mwh \
  --checkpoint_name mwh_ppo_dinov2 \
  --save_every 80 \
  --backbone dinov2 \
  --freeze_backbone


# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/basic_ppo_eval.npz \
  --out ./plots/basic_ppo_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs/mwh_ppo_eval.npz \
  --out ./plots_mwh/mwh_ppo_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs/mwh_ppo_eval_v2.npz \
  --out ./plots_mwh/mwh_ppo_eval_v2.png \
  --annotate_last_only

# 3 output the playing game GIF using ppo agent
python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/basic_ppo_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 320x240 \
  --gif ./out/basic_ppo_v4/best.gif \
  --gif_dir ./out/basic_ppo_v4/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/basic_ppo_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_v5/best.gif \
  --gif_dir ./out/basic_ppo_v5/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints_mwh/mwh_ppo_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_v1/best.gif \
  --gif_dir ./out/mwh_ppo_v1/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints_mwh/mwh_ppo_v2_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_v2/best.gif \
  --gif_dir ./out/mwh_ppo_v2/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic