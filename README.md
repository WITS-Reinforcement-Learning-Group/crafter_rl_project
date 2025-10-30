
# DQN Baseline
python train_dqn.py --env_id CrafterPartial-v1 --total_timesteps 1000000 \
python eval_crafter_metrics.py --model dqn_crafter_baseline.zip --episodes 20 --logdir logs/dqn_csv \
python plot_learning_curve.py --csv logs/dqn_csv/progress.csv --out dqn_base_curve.png \
python crafter_eval_summary.py --path logs/dqn_csv/stats.jsonl --out dqn_summary.png --title "DQN Baseline Evaluation"\

# DQN with Reward Shaping
python train_dqn_RShape.py --env_id CrafterPartial-v1 --total_timesteps 1000000 \
python eval_crafter_metrics.py --model dqn_crafter_rShape.zip --episodes 20 --logdir logs/dqn_rShape_csv \
python plot_learning_curve.py --csv logs/dqn_rShape_csv/progress.csv --out dqn_rShape_curve.png \
python crafter_eval_summary.py --path logs/dqn_rShape_csv/stats.jsonl --out dqn_rShape_summary.png --title "DQN Baseline Reward Shaping Evaluation"

# DQN with N-step Learning
python train_dqn_per_nstep1.py --env_id CrafterPartial-v1 --total_timesteps 1000000 \
python eval_crafter_metrics.py --model dqn_nstep1.zip --episodes 20 --logdir eval_logs/dqn_nstep \
python plot_learning_curve.py --csv logs/dqn_nstep_csv/progress.csv --out dqn_nstep_curve.png \
python crafter_eval_summary.py --path eval_logs/dqn_nstep/stats.jsonl --out dqn_nstep_summary.png --title "DQN Baseline nstep Evaluation"

# DQN with Noisy Networks + N-step
python train_dqn_noisy_nstep.py --env_id CrafterPartial-v1 --total_timesteps 1000000 \
python eval_crafter_metrics.py --model dqn_noisy_nstep.zip --episodes 20 --logdir logs/dqn_noisy_nstep_csv \
python plot_learning_curve.py --csv logs/dqn_noisy_nstep_csv/progress.csv --out dqn_noisy_nstep_curve.png \
python crafter_eval_summary.py --path logs/dqn_noisy_nstep_csv/stats.jsonl --out dqn_noisy_nstep_summary.png --title "DQN Baseline noisy nstep Evaluation"
