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
import collections
import os

numDict = {}
numDict[0] = np.load('nums/0.npy')
numDict[1] = np.load('nums/1.npy')
numDict[2] = np.load('nums/2.npy')
numDict[3] = np.load('nums/3.npy')
numDict[4] = np.load('nums/4.npy')
numDict[5] = np.load('nums/5.npy')
numDict[6] = np.load('nums/6.npy')
numDict[7] = np.load('nums/7.npy')
numDict[8] = np.load('nums/8.npy')
numDict[9] = np.load('nums/9.npy')

unshaped_rewards = {}
shaped_rewards = {}
episode_lengths = {}
generation_achievements = {}

global_last_best_fitness = -9999.0
global_current_sigma = 0.1
STUCK_THRESHOLD = 3.0

try:
    old_gym.register(
        id='CrafterPartial-v1',
        entry_point=crafter.Env,
        disable_env_checker=True
    )
except old_gym.error.Error:
    pass

def extractResourceBar(obs):
    alteredObs = obs.copy()
    blockSize = 7
    health = alteredObs[7*blockSize:7*blockSize+7,0*blockSize:0*blockSize+7].copy()
    food = alteredObs[7*blockSize:7*blockSize+7,1*blockSize:1*blockSize+7].copy()
    water= alteredObs[7*blockSize:7*blockSize+7,2*blockSize:2*blockSize+7].copy()
    stamina = alteredObs[7*blockSize:7*blockSize+7,3*blockSize:3*blockSize+7].copy()
    return health,food,water,stamina

def extractNum(resource):
    alteredResource = resource.copy()
    height, width, _ = resource.shape
    for i in range(height):
        for j in range(width):
            if np.array_equal(resource[i, j], [255, 255, 255]) and i >= 2 and j >= 2:
                alteredResource[i, j] = [255, 255, 255]
            else:
                alteredResource[i, j] = [0, 0, 0]
    return alteredResource

def extractResourceNum(resource):
    for i in range(10):
        if np.array_equal(resource, numDict[i]):
            return i
    return 0

def extractResourceVec(obs):
    health, food, water, stamina = extractResourceBar(obs)
    return [
        extractResourceNum(extractNum(health)),
        extractResourceNum(extractNum(food)),
        extractResourceNum(extractNum(water)),
        extractResourceNum(extractNum(stamina))
    ]

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, num_resources=4):
        super(PolicyNetwork, self).__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        combined_input_size = (32 * 6 * 6) + num_resources
        self.final_layers = nn.Sequential(
            nn.Linear(combined_input_size, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, image_input, resource_input):
        cnn_output = self.cnn_branch(image_input)
        combined = torch.cat((cnn_output, resource_input), dim=1)
        return self.final_layers(combined)

def preprocess_obs(obs):
    obs_np = np.array(obs, dtype=np.float32) / 255.0
    obs_np = np.transpose(obs_np, (2, 0, 1))
    obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
    return obs_tensor

DECREASE_PENALTY = -0.001 
INCREASE_REWARD = 0.001
NOOP_PENALTY = -0.005
INVALID_ACTION_PENALTY = -0.02
ACHIEVEMENT_BONUS = 4.0

def fitness_func(ga_instance, solution, sol_idx):
    global model, torch_ga, args, unshaped_rewards, shaped_rewards, episode_lengths, generation_achievements
    
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution))
    env = old_gym.make('CrafterPartial-v1')
    env = crafter.Recorder(env, f'{args.outdir}/sol_{sol_idx}', save_stats=True, save_video=False, save_episode=False)
    env = GymV21CompatibilityV0(env=env)
    
    total_reward = 0
    unshaped_total_reward = 0
    
    obs, info = env.reset()
    done = False
    
    previous_resources = extractResourceVec(obs)
    previous_inventory = {}

    previous_achievement_count = 0
    
    timestep = 0
    wakeUpDec = False
    collectSaplingDec = False
    while not done:
        timestep = timestep+1
        
        resource_vec = previous_resources
        
        obs_tensor = preprocess_obs(obs)
        resource_tensor = torch.tensor(resource_vec, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = model(obs_tensor, resource_tensor)
        
        action = torch.argmax(action_logits, dim=1).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        unshaped_total_reward += reward
        custom_reward = reward

        current_resources = extractResourceVec(obs)
        current_inventory = info['inventory']

        current_achievements = {ach for ach, count in info['achievements'].items() if count > 0}
        current_achievement_count = len(current_achievements)
        
        custom_reward += ACHIEVEMENT_BONUS*(current_achievement_count-previous_achievement_count)

        if "wake_up" in current_achievements and not wakeUpDec:
            custom_reward += -(ACHIEVEMENT_BONUS*0.7)
            wakeUpDec = True

        if "collect_sapling" in current_achievements and not collectSaplingDec:
            custom_reward += -(ACHIEVEMENT_BONUS*0.7)
            collectSaplingDec = True
        
        previous_achievement_count = current_achievement_count
        
        if action == 0:
            custom_reward += NOOP_PENALTY
        elif 6 <= action <= 16:
            stamina_increased = (current_resources[3] > previous_resources[3])
            inventory_changed = (current_inventory != previous_inventory)
            if action == 6 and not stamina_increased:
                custom_reward += INVALID_ACTION_PENALTY
            elif action >= 7 and not inventory_changed:
                custom_reward += INVALID_ACTION_PENALTY

        if (current_resources[0] < previous_resources[0]): custom_reward += (DECREASE_PENALTY*1.5)*(15-current_resources[0])**2
        if (current_resources[1] < previous_resources[1]): custom_reward += DECREASE_PENALTY*(15-current_resources[1])**2
        if (current_resources[2] < previous_resources[2]): custom_reward += DECREASE_PENALTY*(15-current_resources[2])**2
        if (current_resources[3] < previous_resources[3]): custom_reward += (DECREASE_PENALTY)*(15-current_resources[3])**2
        
        if (current_resources[0] > previous_resources[0]): custom_reward += (INCREASE_REWARD*1.5)*(15-current_resources[0])**2
        if (current_resources[1] > previous_resources[1]): custom_reward += (INCREASE_REWARD)*(15-current_resources[1])**2
        if (current_resources[2] > previous_resources[2]): custom_reward += (INCREASE_REWARD)*(15-current_resources[2])**2
        if (current_resources[3] > previous_resources[3]): custom_reward += (INCREASE_REWARD)*(15-current_resources[3])**2
        
        previous_resources = current_resources
        previous_inventory = current_inventory.copy()
        
        total_reward += custom_reward
        
    env.close()

    unshaped_rewards[sol_idx] = unshaped_total_reward
    shaped_rewards[sol_idx] = total_reward
    episode_lengths[sol_idx] = timestep

    unlocked_set = {ach for ach, count in info['achievements'].items() if count > 0}
    generation_achievements[sol_idx] = unlocked_set
    
    return total_reward

def on_generation_log(ga_instance):
    global unshaped_rewards, shaped_rewards, episode_lengths, generation_achievements
    global global_last_best_fitness, global_current_sigma, STUCK_THRESHOLD
    
    generation = ga_instance.generations_completed
    
    outdir = ga_instance.outdir 
    stats_file_path = os.path.join(outdir, "generation_stats.csv")
    ach_file_path = os.path.join(outdir, "achievement_stats.csv")

    unshaped_scores = list(unshaped_rewards.values())
    shaped_scores = list(shaped_rewards.values())
    lengths = list(episode_lengths.values())
    pop_size = len(generation_achievements)

    avg_unshaped = np.mean(unshaped_scores) if len(unshaped_scores) > 0 else 0.0
    avg_length = np.mean(lengths) if len(lengths) > 0 else 0.0
    
    best_shaped_fitness = np.max(shaped_scores) if len(shaped_scores) > 0 else -9999.0
    avg_shaped_fitness = np.mean(shaped_scores) if len(shaped_scores) > 0 else 0.0
    best_unshaped_fitness = np.max(unshaped_scores) if len(unshaped_scores) > 0 else -9999.0

    new_sigma = global_current_sigma

    fitness_gap = best_shaped_fitness - avg_shaped_fitness
    is_stuck = (fitness_gap < STUCK_THRESHOLD)
    is_regressing = (best_shaped_fitness < global_last_best_fitness)
    
    if is_stuck or is_regressing:
        new_sigma = min(0.5, global_current_sigma * 1.10)
    else:
        new_sigma = max(0.01, global_current_sigma * 0.95)
        
    ga_instance.random_mutation_min_val = -new_sigma
    ga_instance.random_mutation_max_val = new_sigma
    
    global_current_sigma = new_sigma
    global_last_best_fitness = best_shaped_fitness
    
    print(f"Gen {generation: <4} | "
          f"Avg Unshaped R: {avg_unshaped: <8.2f} | "
          f"Avg Shaped R: {avg_shaped_fitness: <8.2f} | "
          f"Avg Length: {avg_length: <8.2f} | "
          f"Max Unshaped R: {best_unshaped_fitness: <8.2f} | "
          f"Max Shaped R: {best_shaped_fitness: <8.2f} | "
          f"Sigma: {new_sigma:.4f}")

    if generation == 1:
        with open(stats_file_path, "w") as f:
            f.write("generation,avg_unshaped_r,avg_shaped_r,avg_length,max_unshaped_r,max_shaped_r,sigma\n")
        with open(ach_file_path, "w") as f:
            f.write("generation,achievement_name,unlock_rate,count,pop_size\n")
    with open(stats_file_path, "a") as f:
        f.write(f"{generation},{avg_unshaped:.4f},{avg_shaped_fitness:.4f},{avg_length:.2f},{best_unshaped_fitness:.4f},{best_shaped_fitness:.4f},{new_sigma:.4f}\n")
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

    unshaped_rewards.clear()
    shaped_rewards.clear()
    episode_lengths.clear()
    generation_achievements.clear()

def on_fitness_select(ga_instance, population_fitness):
    global unshaped_rewards
    
    generation = ga_instance.generations_completed
    
    if (generation + 1) % 5 == 0:
        
        print("--- (Generation %s: Selecting on PURE UNSHAPED reward) ---" % generation)
        
        new_fitness = []
        for i in range(len(population_fitness)):
            new_fitness.append(unshaped_rewards.get(i, -999))
            
        return new_fitness
    else:
        return population_fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='logdir/crafter_ga_imp2/final')
    parser.add_argument('--steps', type=float, default=1e3)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    NUM_ACTIONS = 17
    POPULATION_SIZE = 50
    NUM_GENERATIONS = int(args.steps)
    NUM_PARENTS = 10
    MUTATION_PERCENT = 10

    model = PolicyNetwork(num_actions=NUM_ACTIONS)
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=POPULATION_SIZE)

    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS,
        fitness_func=fitness_func,
        initial_population=torch_ga.population_weights,
        mutation_percent_genes=MUTATION_PERCENT,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_by_replacement=False,
        random_mutation_min_val=-global_current_sigma,
        random_mutation_max_val=global_current_sigma,
        on_fitness=on_fitness_select,
        on_generation=on_generation_log
    )

    ga_instance.outdir = args.outdir

    print(f"--- Starting GA Evolution for {NUM_GENERATIONS} generations ---")
    print(f"--- Logging results to {args.outdir} ---")
    ga_instance.run()
    print("--- Evolution Finished ---")

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution fitness: {solution_fitness}")
    
    best_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(best_weights_dict)
    torch.save(model.state_dict(), f"{args.outdir}/best_ga_policy.pth")
    print(f"Best model weights saved to {args.outdir}/best_ga_policy.pth")

    ga_instance.plot_fitness()