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

try:
    old_gym.register(
        id='CrafterPartial-v1',
        entry_point=crafter.Env,
        disable_env_checker=True
    )
except old_gym.error.Error:
    pass

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

def fitness_func(ga_instance, solution, sol_idx):
    global model, torch_ga, args
    
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
    
    while not done:
        obs_tensor = preprocess_obs(obs)
        
        with torch.no_grad():
            action_logits = model(obs_tensor)
        
        action = torch.argmax(action_logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    env.close()
    return total_reward

def on_generation_callback(ga_instance):
    generation = ga_instance.generations_completed
    best_fitness = ga_instance.best_solution()[1]
    pop_fitness = ga_instance.last_generation_fitness
    avg_fitness = np.mean(pop_fitness)
    print(f"Generation {generation: <4} | Best Fitness: {best_fitness: <8.2f} | Avg Fitness: {avg_fitness: <8.2f}")



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
    on_generation=on_generation_callback
)

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
