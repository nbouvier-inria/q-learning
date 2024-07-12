"""
A script that runs the NEVA algorithm to
train spiking neural networks using 
multithreading. The simulation runs 
on most discrete action spaces from
gymnasium.
"""

from collections import namedtuple
import numpy as np
from spikingjelly.activation_based import functional
import math
import gymnasium as gym
from tqdm import tqdm
from tools import DQSN, torus, graph_to_N, combine_tensors_1point, mutate_tensor, torus_no_comm, bit_flips
import torch
from multiprocessing.pool import Pool, ThreadPool

""" Hyperparameters """
# General simulation control
GENERATIONS = 100
THREADS = 40
ENV_NAME = "CartPole-v1"
SAVING_PERIOD = 5

# Neural network architecture
HIDDEN_LAYERS = 1
LAYERS_SIZE = 25
VTH = 1.
T = 16

# Genetic topology
N = 49
v, e = torus(N)
NEIGHBOURS = graph_to_N(e, v)
COMBINE = lambda x, y: combine_tensors_1point(x, y)
MUTATE = lambda x : mutate_tensor(x, 1)
P_COMBINE = 0.5
P_MUTATE = 0.5

""" Parameters """
ENV_NAME = "CartPole-v1"
    
""" Learning scripts """

def get_reward(policy: DQSN):
    """ Returns the reward of the policy """
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
    """ Returns the best policies for
    the problem stated in the global
    parameters """
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    population = [DQSN(input_size=n_states, output_size=n_actions, hidden_layers=HIDDEN_LAYERS, hidden_size=LAYERS_SIZE, Vth=VTH, T=T).to(torch.device("cuda")) for _ in range(N)]
    
    # Modified version of to_tensor, from_tensor and
    # get_reward that work on lists of tensors
    tensorize = lambda x: [y.to_tensor() for y in x]
    def rewardize(population):
        with ThreadPool(THREADS) as p:
            rewards = np.array(list(p.map(get_reward, population)))
        return rewards
    untensorize = lambda x, w : [y.from_tensor(t) for y, t in zip(x, w)]

    # Set the weights of the neural networks to binary values
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
                new_weights[i] = MUTATE(COMBINE(weights[np.random.choice(better_neighbours)], weights[i]))
            else:
                new_weights[i] = MUTATE(weights[i])
    
        untensorize(population, new_weights)
        new_rewards = rewardize(population)
        return new_weights, new_rewards 

    # Initialize rewards and weights
    rewards = rewardize(population)
    weights = tensorize(population)
    best_rewards = rewards
    best_weights = weights
    average_rewards = np.zeros_like(rewards)

    for i in tqdm(range(GENERATIONS)):
        # Update the sliding average
        gamma = 0
        average_rewards = (rewards + average_rewards * gamma)/(1+gamma)
        print(np.reshape(average_rewards.round(), newshape=(int(np.sqrt(N)), int(np.sqrt(N)))))

        # Update the weights and best solutions
        new_weights, new_rewards = update_weights(weights, rewards, population)
        best_weights = [new_weights[i] if new_rewards[i] >= best_rewards[i] else best_weights[i] for i in range(len(new_weights))]
        best_rewards = np.maximum(new_rewards, best_rewards)
        weights = [nw if nr > r else w for r, nr, w, nw in zip(average_rewards, new_rewards, weights, new_weights)]

        # Print informations
        np.reshape(a=np.where(( new_rewards) > average_rewards, 
                                    [f"n:{int(r)}" for r in new_rewards],
                                    [f"o:{int(r)}" for r in rewards]), 
                                    newshape=(int(np.sqrt(N)), int(np.sqrt(N))))
        print(f"Bit-to-bit average difference:  {round(np.average([bit_flips(x,y) for x in weights for y in weights]))}")
        
        # Compute the new rewards
        rewards = rewardize(population)
        untensorize(population, weights)

        # Print the average reward
        med = np.median(rewards)
        print(f"Median of rewards and averaged rewards are {med} and {np.median(average_rewards)}")

        # Reset to personnal best every few generations
        if i % SAVING_PERIOD == 0:
            weights = [weights[i] if rewards[i] >= med or best_rewards[i] <= med else best_weights[i] for i in range(len(weights))]
    return best_weights
        

""" Run the script """
if __name__ == "__main__":
    nevai()