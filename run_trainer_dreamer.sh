python -m scripts.train_dreamerv2 \
  --scenario basic \
  --total_iterations 200 \
  --steps_per_iteration 256\
  --tb_log_dir ./logs/tb_dreamerv2_basic \
  --eval_log_dir ./logs \
  --eval_log_name dreamerv2_basic_run1_eval.npz \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name dreamerv2_basic_run1 \
  --save_every 0