import gym as old_gym
import crafter
import torch
import torch.nn as nn
import numpy as np
import pygad
import pygad.torchga
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import argparse
import collections  # <-- NEW
import os           # <-- NEW

try:
    old_gym.register(
        id='CrafterPartial-v1',
        entry_point=crafter.Env,
        disable_env_checker=True
    )
except old_gym.error.Error:
    pass

# --- NEW: Global dictionaries for logging ---
unshaped_rewards = {}
shaped_rewards = {}
episode_lengths = {}
generation_achievements = {}
# --- END NEW ---

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),     
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def preprocess_obs(obs):
    obs_np = np.array(obs, dtype=np.float32) / 255.0
    obs_np = np.transpose(obs_np, (2, 0, 1))
    obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
    return obs_tensor

# --- MODIFIED: fitness_func now logs all stats ---
def fitness_func(ga_instance, solution, sol_idx):
    global model, torch_ga, args
    # --- NEW: Get logging dicts ---
    global unshaped_rewards, shaped_rewards, episode_lengths, generation_achievements
    
    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model,
        weights_vector=solution
    )
    model.load_state_dict(model_weights_dict)

    env = old_gym.make('CrafterPartial-v1')
    
    env = crafter.Recorder(
      env,
      f'{args.outdir}/sol_{sol_idx}', 
      save_stats=True,
      save_video=False,
      save_episode=False,
    )

    env = GymV21CompatibilityV0(env=env)
    
    total_reward = 0
    obs, info = env.reset()
    done = False
    
    timestep = 0  # <-- NEW
    while not done:
        timestep += 1 # <-- NEW
        obs_tensor = preprocess_obs(obs)
        
        with torch.no_grad():
            action_logits = model(obs_tensor)
        
        action = torch.argmax(action_logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    env.close()
    
    # --- NEW: Store all stats for logging ---
    unshaped_rewards[sol_idx] = total_reward
    shaped_rewards[sol_idx] = total_reward # For a baseline, shaped == unshaped
    episode_lengths[sol_idx] = timestep
    unlocked_set = {ach for ach, count in info['achievements'].items() if count > 0}
    generation_achievements[sol_idx] = unlocked_set
    # --- END NEW ---
    
    return total_reward

# --- REPLACED: New advanced logging callback ---
def on_generation_log(ga_instance):
    global unshaped_rewards, shaped_rewards, episode_lengths, generation_achievements
    
    generation = ga_instance.generations_completed
    
    # --- 1. Define file paths ---
    outdir = ga_instance.outdir 
    stats_file_path = os.path.join(outdir, "generation_stats.csv")
    ach_file_path = os.path.join(outdir, "achievement_stats.csv")

    # --- 2. Calculate Averages & Max ---
    unshaped_scores = list(unshaped_rewards.values())
    shaped_scores = list(shaped_rewards.values())
    lengths = list(episode_lengths.values())
    pop_size = len(generation_achievements)

    avg_unshaped = np.mean(unshaped_scores) if len(unshaped_scores) > 0 else 0.0
    avg_length = np.mean(lengths) if len(lengths) > 0 else 0.0
    
    best_shaped_fitness = np.max(shaped_scores) if len(shaped_scores) > 0 else -9999.0
    avg_shaped_fitness = np.mean(shaped_scores) if len(shaped_scores) > 0 else 0.0
    best_unshaped_fitness = np.max(unshaped_scores) if len(unshaped_scores) > 0 else -9999.0

    # --- 3. Print the console log ---
    print(f"Gen {generation: <4} | "
          f"Avg Unshaped R: {avg_unshaped: <8.2f} | "
          f"Avg Shaped R: {avg_shaped_fitness: <8.2f} | "
          f"Avg Length: {avg_length: <8.2f} | "
          f"Max Unshaped R: {best_unshaped_fitness: <8.2f} | "
          f"Max Shaped R: {best_shaped_fitness: <8.2f}")

    # --- 4. Write CSV Headers (only on first generation) ---
    if generation == 1:
        with open(stats_file_path, "w") as f:
            f.write("generation,avg_unshaped_r,avg_shaped_r,avg_length,max_unshaped_r,max_shaped_r\n")
        with open(ach_file_path, "w") as f:
            f.write("generation,achievement_name,unlock_rate,count,pop_size\n")

    # --- 5. Append data to CSV files ---
    with open(stats_file_path, "a") as f:
        f.write(f"{generation},{avg_unshaped:.4f},{avg_shaped_fitness:.4f},{avg_length:.2f},{best_unshaped_fitness:.4f},{best_shaped_fitness:.4f}\n")
    
    if pop_size > 0:
        ach_counts = collections.Counter()
        for ach_set in generation_achievements.values():
            ach_counts.update(ach_set)
        
        sorted_counts = sorted(ach_counts.items(), key=lambda item: item[1], reverse=True)
        
        with open(ach_file_path, "a") as f:
            for ach, count in sorted_counts:
                rate = (count / pop_size)
                print(f"    - {ach:<20}: {rate*100:.1f}% ({count}/{pop_size})")
                f.write(f"{generation},{ach},{rate:.4f},{count},{pop_size}\n")

    # --- 6. CRITICAL: Clear all dictionaries ---
    unshaped_rewards.clear()
    shaped_rewards.clear()
    episode_lengths.clear()
    generation_achievements.clear()
# --- END REPLACED ---


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_ga_base/final')
parser.add_argument('--steps', type=float, default=1e3)
args = parser.parse_args()

torch.set_grad_enabled(False)

NUM_ACTIONS = 17       
POPULATION_SIZE = 50   
NUM_GENERATIONS = int(args.steps)
NUM_PARENTS = 10       
MUTATION_PERCENT = 10 

model = PolicyNetwork(num_actions=NUM_ACTIONS)
torch_ga = pygad.torchga.TorchGA(
    model=model,
    num_solutions=POPULATION_SIZE
)

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS,
    fitness_func=fitness_func,
    initial_population=torch_ga.population_weights, 
    mutation_percent_genes=MUTATION_PERCENT,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    on_generation=on_generation_log  # <-- CHANGED
)

# --- NEW: Attach outdir to the instance ---
ga_instance.outdir = args.outdir
# --- END NEW ---

print(f"--- Starting GA Evolution for {NUM_GENERATIONS} generations ---")
print(f"--- Logging results to {args.outdir} ---")

ga_instance.run()

print("--- Evolution Finished ---")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution fitness: {solution_fitness}")

best_weights_dict = pygad.torchga.model_weights_as_dict(
    model=model,
    weights_vector=solution
)
model.load_state_dict(best_weights_dict)
torch.save(model.state_dict(), f"{args.outdir}/best_ga_policy.pth")
print(f"Best model weights saved to {args.outdir}/best_ga_policy.pth")

ga_instance.plot_fitness()
