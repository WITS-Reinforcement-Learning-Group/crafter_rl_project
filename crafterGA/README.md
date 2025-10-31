# 🧬 Crafter GA Experiments

## 🚀 Setup

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate crafter_env_ga
```

## 🏃 Training

**Baseline Training (200 generations):**
```bash
python train_ga.py --outdir logdir/crafter_ga_base/final --steps 200
```

**Improvement 1 Training (200 generations):**
```bash
python train_ga_surv_v3.py --outdir logdir/crafter_ga_imp2/final --steps 200
```

**Improvement 2 Training (200 generations):**
```bash
python train_ga_surv_v4.py --outdir logdir/crafter_ga_imp3/final --steps 200
```

## 📊 Evaluation

**Baseline Evaluation:**
```bash
python pygadEval.py --model_path logdir/crafter_ga_base/final/best_ga_policy.pth --outdir logdir/crafter_ga_eval/base --seed 42
```

**Improvment 1 Evaluation:**
```bash
python pygadEvalImp1.py --model_path logdir/crafter_ga_imp1/final/best_ga_policy.pth --outdir logdir/crafter_ga_eval/imp2/final --seed 42
```

**Improvment 2 Evaluation:**
```bash
python pygadEvalImp1.py --model_path logdir/crafter_ga_imp2/final/best_ga_policy.pth --outdir logdir/crafter_ga_eval/imp3/final --seed 42
```

**Animation Creation:**
```bash
python viewEpisode.py --filename logdir/crafter_ga_eval/base/episode.npz
```
```bash
python viewEpisode.py --filename logdir/crafter_ga_eval/imp2/final/episode.npz
```
```bash
python viewEpisode.py --filename logdir/crafter_ga_eval/imp3/final/episode.npz
```

**Graph Creation:**
```bash
python pygadPlots.py
```


## 📁 Project Structure
```
.
├── environment.yml              # Conda environment
├── train_ga.py                  # Baseline
├── train_ga_surv_v3.py          # Improvement 1
├── train_ga_surv_v4.py          # Improvement 2
└── logdir/                      # Outputs
```

## 📚 References

- [Crafter Benchmark](https://github.com/danijar/crafter)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyGAD](https://pygad.readthedocs.io/en/latest/index.html)

---

**Happy Training! 🚀**
