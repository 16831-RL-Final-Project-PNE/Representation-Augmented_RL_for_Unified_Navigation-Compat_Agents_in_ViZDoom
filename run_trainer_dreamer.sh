python -m scripts.train_dreamerv2 \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 16 \
  --tb_log_dir ./logs/tb_dreamerv2_basic_2 \
  --eval_log_dir ./logs/basic_2 \
  --eval_log_name dreamerv2_basic_run2_eval.npz \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name dreamerv2_basic_run2 \
  --save_every 30

# python -m scripts.train_dreamerv2 \
#   --scenario my_way_home \
#   --action_space usual \
#   --total_iterations 200 \
#   --steps_per_iteration 8192 \
#   --tb_log_dir ./logs/tb_dreamerv2_my_way_home \
#   --eval_log_dir ./logs/my_way_home \
#   --eval_log_name dreamerv2_my_way_home_run1_eval.npz \
#   --checkpoint_dir ./checkpoints \
#   --checkpoint_name dreamerv2_my_way_home_run1 \
#   --save_every 30