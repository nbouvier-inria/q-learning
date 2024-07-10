from collections import namedtuple
import numpy as np
from spikingjelly.activation_based import functional
import math
import gymnasium as gym
from tqdm import tqdm
from tools import DQSN, torus, graph_to_N, combine_tensors_1point, mutate_tensor
import torch
from multiprocessing.pool import Pool, ThreadPool

""" Hyperparameters """
# General simulation control
GENERATIONS = 100
THREADS = 40
ENV_NAME = "CartPole-v1"

# Neural network architecture (Unused)
HIDDEN_LAYERS = 2
LAYERS_SIZE = 50
VTH = 1.
T = 16 

# Genetic topology
N = 49
v, e = torus(N)
NEIGHBOURS = graph_to_N(e, v)
MUTATION_STRENGTH = 1/1000
COMBINE = combine_tensors_1point
MUTATE = mutate_tensor
P_COMBINE = 0.5
P_MUTATE = 0.5

""" Parameters """
ENV_NAME = "CartPole-v1"
    
""" Learning scripts """

def get_reward(policy: DQSN):
    env = gym.make(ENV_NAME)
    env.reset()
    n_states = env.observation_space.shape[0]
    state = torch.zeros(size=[1, n_states], dtype=torch.float, device=torch.device("cuda"))
    done = False
    total_reward = 0
    while not done:
        action = policy(state).max(1).indices
        action = action.view(1, 1)
        functional.reset_net(policy)
        state, reward, termination, truncation, _ = env.step(action.item())
        done = termination or truncation
        total_reward += reward
        state = torch.from_numpy(state).float().to(torch.device("cuda")).unsqueeze(0)
    env.close()
    return total_reward

def nevai():
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    population = [DQSN(input_size=n_states, output_size=n_actions, hidden_layers=HIDDEN_LAYERS, hidden_size=LAYERS_SIZE, Vth=VTH, T=T).to(torch.device("cuda")) for _ in range(N)]
    
    tensorize = lambda x: [y.to_tensor() for y in x]
    def rewardize(population):
        with ThreadPool(THREADS) as p:
            rewards = np.array(list(p.map(get_reward, population)))
        return rewards
    untensorize = lambda x, w : [y.from_tensor(t) for y, t in zip(x, w)]

    t = tensorize(population)
    t = [torch.randint_like(input=tensor,low=-1, high=2) for tensor in t]
    untensorize(population, t)

    def update_weights(weights, rewards, population):
        """ Sets weights using an evolutionnary
        algorithm based on rewards gained """
        new_weights = [0 for _ in range(N)]
        for i in range(N):
            better_neighbours = []
            for j in NEIGHBOURS[i]:
                if rewards[j] > rewards[i]:
                    better_neighbours.append(j)
            if better_neighbours:
                new_weights[i] = mutate_tensor(combine_tensors_1point(weights[np.random.choice(better_neighbours)], weights[i]), 1)
            else:
                new_weights[i] = mutate_tensor(weights[i], 1)
        untensorize(population, new_weights)
        new_rewards = rewardize(population)
        return new_weights, new_rewards 

    rewards = rewardize(population)
    weights = tensorize(population)
    for _ in tqdm(range(GENERATIONS)): 
        new_weights, new_rewards = update_weights(weights, rewards, population)
        weights = [nw if nr > r else w for r, nr, w, nw in zip(rewards, new_rewards, weights, new_weights)]
        print(np.where(new_rewards > rewards, new_rewards, "Unchanged"))
        # rewards = np.maximum(rewards, new_rewards)
        rewards = rewardize(population)
        untensorize(population, weights)
        

""" Run the script """
if __name__ == "__main__":
    nevai()