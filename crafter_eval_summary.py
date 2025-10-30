#!/usr/bin/env python3
import argparse, json, math, os, glob, csv, statistics as stats
import matplotlib.pyplot as plt

ACHIEVEMENTS = [
    "achievement_collect_wood", "achievement_collect_stone", "achievement_collect_iron", "achievement_collect_coal", "achievement_collect_diamond",
    "achievement_collect_drink", "achievement_collect_sapling",
    "achievement_place_table", "achievement_place_furnace", "achievement_place_plant", "achievement_place_stone",
    "achievement_make_wood_pickaxe", "achievement_make_stone_pickaxe", "achievement_make_iron_pickaxe",
    "achievement_make_wood_sword", "achievement_make_stone_sword", "achievement_make_iron_sword",
    "achievement_eat_cow", "achievement_eat_plant",
    "achievement_defeat_zombie", "achievement_defeat_skeleton",
    "achievement_wake_up"
]

# Common key aliases seen in different Crafter/eval loggers
LEN_CANDIDATES = ["episode_len", "length", "steps", "survival_time", "t", "n_steps"]
RET_CANDIDATES = ["episode_return", "return", "cumulative_reward", "reward", "episode_reward", "sum_reward"]

def load_episode_lines(path):
    # path can be a stats.json (newline-delimited) or a folder with many episode_*.json files
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "episode_*.json")))
        for f in files:
            with open(f, "r") as fh:
                yield json.load(fh)
    else:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

def get_first_present_key(d, candidates, default=None):
    for k in candidates:
        if k in d:
            return k
    return default

def extract_episode_len(ep):
    # direct or nested
    key = get_first_present_key(ep, LEN_CANDIDATES)
    if key is not None:
        return ep.get(key)
    # look inside 'info' or similar nested dicts
    for nest in ["info", "episode", "metrics"]:
        if isinstance(ep.get(nest), dict):
            key = get_first_present_key(ep[nest], LEN_CANDIDATES)
            if key is not None:
                return ep[nest].get(key)
    return None

def extract_episode_return(ep):
    key = get_first_present_key(ep, RET_CANDIDATES)
    if key is not None:
        return ep.get(key)
    for nest in ["info", "episode", "metrics"]:
        if isinstance(ep.get(nest), dict):
            key = get_first_present_key(ep[nest], RET_CANDIDATES)
            if key is not None:
                return ep[nest].get(key)
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="eval_logs/.../stats.json OR a folder with episode_*.json files")
    ap.add_argument("--out", default="crafter_eval_summary.png", help="Output PNG for the achievements plot")
    ap.add_argument("--title", default="Crafter Evaluation Summary", help="Plot title")
    ap.add_argument("--csv", default="", help="Optional CSV path to save per-episode metrics")
    args = ap.parse_args()

    episodes = list(load_episode_lines(args.path))
    n = len(episodes)
    assert n > 0, "No episodes found"

    # ---------- Achievement unlock rates ----------
    # Compute unlock rates (>=1 occurrence in an episode)
    rates = {}
    for k in ACHIEVEMENTS:
        c = sum(1 for ep in episodes if ep.get(k, 0) > 0)
        rates[k] = c / n

    # ---------- Crafter Score (geometric mean of unlock rates) ----------
    epsilon = 1e-9  # protects against log(0)
    geo_sum = 0.0
    for k in ACHIEVEMENTS:
        geo_sum += math.log(max(rates[k], epsilon))
    crafter_score = math.exp(geo_sum / len(ACHIEVEMENTS))

    # ---------- Survival time & cumulative reward ----------
    lengths = []
    returns = []
    missing_len, missing_ret = 0, 0
    for ep in episodes:
        L = extract_episode_len(ep)
        R = extract_episode_return(ep)
        if L is None: missing_len += 1
        else: lengths.append(float(L))
        if R is None: missing_ret += 1
        else: returns.append(float(R))

    def summarize(vals, name):
        if len(vals) == 0:
            return f"{name}: N/A (no values found)"
        m = stats.mean(vals)
        sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
        return f"{name}: mean={m:.3f}, std={sd:.3f}, min={min(vals):.3f}, max={max(vals):.3f}, n={len(vals)}"

    # ---------- Console output ----------
    print(f"Episodes: {n}")
    print(f"Crafter Score: {crafter_score:.4f} ({crafter_score:.3e})")
    print("\nAchievement unlock rate (%):")
    for k, v in sorted(rates.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:30s} {100.0*v:6.2f}%")

    print()
    print(summarize(lengths, "Survival time (timesteps/episode)"))
    if missing_len:
        print(f"  Note: {missing_len} episodes missing a recognized length key ({', '.join(LEN_CANDIDATES)})")

    print(summarize(returns, "Cumulative reward (per episode)"))
    if missing_ret:
        print(f"  Note: {missing_ret} episodes missing a recognized return key ({', '.join(RET_CANDIDATES)})")

    # ---------- Optional CSV export ----------
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            header = ["episode_index", "episode_len", "episode_return"] + ACHIEVEMENTS
            w.writerow(header)
            for i, ep in enumerate(episodes):
                row = [
                    i,
                    extract_episode_len(ep),
                    extract_episode_return(ep),
                ] + [ep.get(k, 0) for k in ACHIEVEMENTS]
                w.writerow(row)
        print(f"\nSaved per-episode metrics CSV -> {args.csv}")

    # ---------- Plot top-K achievements ----------
    top = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [k.replace("achievement_", "") for k,_ in top]
    vals = [v for _,v in top]

    plt.figure(figsize=(10,5))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.ylim(0,1.0)
    plt.ylabel("Unlock rate")
    plt.title(f"{args.title}  |  Crafter Score: {crafter_score:.3f}  (n={n})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot -> {args.out}")
