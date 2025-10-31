# Crafter RL Experiments

A collection of deep reinforcement learning experiments on the [Crafter](https://github.com/danijar/crafter) environment, featuring both DQN and PPO implementations with various improvements.

## 🎮 About Crafter

Crafter is a challenging open-world survival game designed for benchmarking RL agents. It requires learning complex behaviors like resource gathering, crafting, and combat across diverse terrains.

## 📋 Requirements

- Python 3.10
- CUDA-capable GPU (optional, but recommended for faster training)
- Conda or Miniconda

## 🚀 Setup

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate crafter_env
```

### 2. Verify Installation
```bash
python -c "import crafter; import stable_baselines3; print('Setup successful!')"
```

### Optional: GA Conda Environment
```bash
conda env create -f crafterGA/environment.yml
conda activate crafter_env_ga
```

## 🏃 Training

### PPO Training (1M steps each)

**Baseline PPO:**
```bash
python train.py --outdir logdir/ppo_baseline --steps 1000000
```

**Improved PPO:**
```bash
python train2.py --outdir logdir/ppo_improved --steps 1000000
```

**PPO with Curiosity:**
```bash
python train3.py --outdir logdir/ppo_curiosity --steps 1000000
```

### DQN Training (1M steps each)

**Baseline DQN:**
```bash
python train_dqn.py --env_id CrafterPartial-v1 --total_timesteps 1000000
```

**DQN with Reward Shaping:**
```bash
python train_dqn_RShape.py --env_id CrafterPartial-v1 --total_timesteps 1000000
```

**DQN with N-step Learning:**
```bash
python train_dqn_per_nstep1.py --env_id CrafterPartial-v1 --total_timesteps 1000000
```

**DQN with Noisy Networks + N-step:**
```bash
python train_dqn_noisy_nstep.py --env_id CrafterPartial-v1 --total_timesteps 1000000
```
### GA Training (200 generations each)

**Baseline Training:**
```bash
python crafterGA/train_ga.py --outdir crafterGA/logdir/crafter_ga_base/final --steps 200
```

**Improvement 1 Training:**
```bash
python crafterGA/train_ga_surv_v3.py --outdir crafterGA/logdir/crafter_ga_imp2/final --steps 200
```

**Improvement 2 Training:**
```bash
python crafterGA/train_ga_surv_v4.py --outdir crafterGA/logdir/crafter_ga_imp3/final --steps 200
```

## 📊 Evaluation

### DQN Evaluation

**Baseline DQN:**
```bash
python eval_crafter_metrics.py --model dqn_crafter_baseline.zip --episodes 20 --logdir logs/dqn_csv
python plot_learning_curve.py --csv logs/dqn_csv/progress.csv --out dqn_base_curve.png
python crafter_eval_summary.py --path logs/dqn_csv/stats.jsonl --out dqn_summary.png --title "DQN Baseline Evaluation"
```

**DQN with Reward Shaping:**
```bash
python eval_crafter_metrics.py --model dqn_crafter_rShape.zip --episodes 20 --logdir logs/dqn_rShape_csv
python plot_learning_curve.py --csv logs/dqn_rShape_csv/progress.csv --out dqn_rShape_curve.png
python crafter_eval_summary.py --path logs/dqn_rShape_csv/stats.jsonl --out dqn_rShape_summary.png --title "DQN Reward Shaping Evaluation"
```

**DQN with N-step Learning:**
```bash
python eval_crafter_metrics.py --model dqn_nstep1.zip --episodes 20 --logdir eval_logs/dqn_nstep
python plot_learning_curve.py --csv logs/dqn_nstep_csv/progress.csv --out dqn_nstep_curve.png
python crafter_eval_summary.py --path eval_logs/dqn_nstep/stats.jsonl --out dqn_nstep_summary.png --title "DQN N-step Evaluation"
```

**DQN with Noisy Networks + N-step:**
```bash
python eval_crafter_metrics.py --model dqn_noisy_nstep.zip --episodes 20 --logdir logs/dqn_noisy_nstep_csv
python plot_learning_curve.py --csv logs/dqn_noisy_nstep_csv/progress.csv --out dqn_noisy_nstep_curve.png
python crafter_eval_summary.py --path logs/dqn_noisy_nstep_csv/stats.jsonl --out dqn_noisy_nstep_summary.png --title "DQN Noisy + N-step Evaluation"
```

### GA Evaluation
**Baseline Evaluation:**
```bash
python crafterGA/pygadEval.py --model_path crafterGA/logdir/crafter_ga_base/final/best_ga_policy.pth --outdir crafterGA/logdir/crafter_ga_eval/base --seed 42
```

**Improvment 1 Evaluation:**
```bash
python crafterGA/pygadEvalImp1.py --model_path crafterGA/logdir/crafter_ga_imp1/final/best_ga_policy.pth --outdir crafterGAlogdir/crafter_ga_eval/imp2/final --seed 42
```

**Improvment 2 Evaluation:**
```bash
python crafterGA/pygadEvalImp1.py --model_path crafterGA/logdir/crafter_ga_imp2/final/best_ga_policy.pth --outdir crafterGA/logdir/crafter_ga_eval/imp3/final --seed 42
```

**Animation Creation:**
```bash
python crafterGA/viewEpisode.py --filename crafterGA/logdir/crafter_ga_eval/base/episode.npz
```
```bash
python crafterGA/viewEpisode.py --filename crafterGA/logdir/crafter_ga_eval/imp2/final/episode.npz
```
```bash
python crafterGA/viewEpisode.py --filename crafterGA/logdir/crafter_ga_eval/imp3/final/episode.npz
```

**Graph Creation:**
```bash
python crafterGA/pygadPlots.py
```

## 🔍 Monitoring Training

Monitor PPO training with TensorBoard:
```bash
tensorboard --logdir logdir/
```

## 📁 Project Structure
```
.
├── environment.yml              # Conda environment
├── train.py                     # Baseline PPO
├── train2.py                    # Improved PPO
├── train3.py                    # PPO + Curiosity
├── train_dqn.py                 # Baseline DQN
├── train_dqn_RShape.py          # DQN + Reward Shaping
├── train_dqn_per_nstep1.py      # DQN + N-step
├── train_dqn_noisy_nstep.py     # DQN + Noisy + N-step
├── crafterGA/                   # GA Files
└── logdir/                      # Training outputs
```

## 📚 References

- [Crafter Benchmark](https://github.com/danijar/crafter)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyGAD](https://pygad.readthedocs.io/en/latest/index.html)

---

**Happy Training! 🚀**
