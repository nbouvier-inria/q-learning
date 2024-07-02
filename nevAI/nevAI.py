import gymnasium as gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from spikingjelly.activation_based import monitor, neuron, functional, layer
import os
from random import choice


def torus(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Square orus-shaped graph (V, E) of
    less than n vertices
    """
    n = int(np.sqrt(n))
    V = [i for i in range(n**2)]
    E = []
    for i in range(n - 1):
        E.append((i + n * (n - 1), i + 1 + n * (n - 1)))
        E.append((i + n * (n - 1), i))
        E.append((n - 1 + n * (i), n - 1 + n * (i + 1)))
        E.append((n - 1 + n * (i), 1 + n * (i)))
        for j in range(n - 1):
            E.append((i + n * j, i + n * j + 1))
            E.append((i + n * j, i + n * (j + 1)))
    return V, E

def graph_to_N(E, V):
    """
    Returns the non-directed neighbourhood based
    on graph G = (V, E)
    """
    N = [[] for _ in V]
    for i, j in E:
        N[i].append(j)
        N[j].append(i)
    return N


def combine_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    """
    Combine two tensors of the same shape, where each element of the resulting tensor
    has a 50% chance of coming from either of the input tensors.
    Args:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor
    Returns:
        torch.Tensor: Combined tensor
    """
    mask = torch.rand_like(a) > 0.5  # generate a random mask
    return torch.where(mask, a, b)


def bit_flips(tensor1, tensor2):
    """
    Calculate the number of bit flips required to get from tensor1 to tensor2.
    Args:
        tensor1 (torch.Tensor): Binary tensor (0s and 1s)
        tensor2 (torch.Tensor): Binary tensor (0s and 1s)
    Returns:
        int: Number of bit flips required
    """
    # Ensure tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    # Calculate the XOR of the two tensors (1 if bits are different, 0 if same)
    xor_tensor = torch.logical_xor(tensor1, tensor2).int()
    # Sum the number of 1s in the XOR tensor (each 1 represents a bit flip)
    num_flips = xor_tensor.sum().item()
    return num_flips

import torch

def voltage_modify(x, T):
        x = 2*x/T
        return torch.where(x > 1, x - 1, -1/x)


def mutate_tensor(tensor, k):
    """
    Mutate a binary tensor by flipping k random bits.
    Args:
        tensor (torch.Tensor): Binary tensor (0s and 1s)
        k (int): Number of bits to flip
    Returns:
        torch.Tensor: Mutated tensor
    """
    # Get the shape of the tensor
    num_elements = tensor.numel()
    # Check if k is not larger than the number of elements in the tensor
    assert k <= num_elements, "k cannot be larger than the number of elements in the tensor"
    # Create a tensor with k random indices to flip
    indices = torch.randperm(num_elements)[:k]
    result = tensor.clone()
    for idx in indices:
        # Randomly select a value to flip to (-1, 0, or 1)
        value = torch.randint(0, 2, (1,)).item()
        result.view(-1)[idx] = value
    return result

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)

        if self.training:
            self.neuronal_charge(x)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
                else:
                    self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
                
            else:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
                else:
                    self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)


# Spiking DQN algorithm
class DQSN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=1, T=16):
        super().__init__()

        layers = OrderedDict()
        layers["0th_layer"] = layer.Linear(input_size, hidden_size)
        layers["0th_IF"] = neuron.IFNode()
        for i in range(hidden_layers):
            layers[f"{i+1}th_laye"] = layer.Linear(hidden_size, hidden_size)
            layers[f"{i+1}th_IF"] = neuron.IFNode()
        layers[f"{hidden_layers+1}th_layer"] = layer.Linear(hidden_size, output_size)
        layers[f"output_layer"] = NonSpikingLIFNode(tau=2.0)

        self.fc = nn.Sequential(
            layers
        )

        self.T = T

    def forward(self, x):
        for t in range(self.T):
            self.fc(x)
            
        return voltage_modify(self.fc[-1].v, self.T)


def train(use_cuda, model_dir, log_dir, env_name, hidden_size, num_episodes, seed):
    BATCH_SIZE = 12
    REPLAY_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 1
    HIDDEN_LAYERS = 5
    N = 16
    v, e = torus(N)
    NEIGHBOURS = graph_to_N(e, v)

    GAMMA = 1/100
    T = 16

    
    def set_binary():
        lengths = [[policy_net[individual].fc[2*i].weight.flatten() for i in range(HIDDEN_LAYERS+2)] for individual in range(N)]
        weights = [torch.cat(length) for length in lengths]
        dimension = weights[0].shape
        weights = [torch.randint(0, 2, dimension).float().to(device) for _ in weights]
        
        for individual in range(N):
        # Update the weights to the new computed values if a change have been made
            lengths[individual] = [i.size(dim=0) for i in lengths[individual]]
            cursor = 0
            for i in range(HIDDEN_LAYERS+2):
                shape = policy_net[individual].fc[2*i].weight.size()
                length = lengths[individual][i]
                policy_net[individual].fc[2*i].weight = torch.nn.Parameter(data=weights[individual][cursor:(cursor+length)].view(shape), requires_grad=False)
                cursor += length

    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")

    steps_done = 0

    env = [gym.make(env_name, render_mode="rgb_array").unwrapped for _ in range(N)]

    n_states = env[0].observation_space.shape[0]
    n_actions = env[0].action_space.n

    # Create spiking neural networks
    policy_net: List[DQSN]= [DQSN(input_size=n_states, hidden_size=hidden_size, output_size=n_actions, hidden_layers=HIDDEN_LAYERS, T=T).to(device) for _ in range(N)]
    target_net: DQSN = DQSN(input_size=n_states, hidden_size=hidden_size, output_size=n_actions, hidden_layers=HIDDEN_LAYERS, T=T).to(device) # for _ in range(N)]
    set_binary()
    target_net.load_state_dict(policy_net[0].state_dict()) # (target_net[i].load_state_dict(policy_net[i].state_dict()) for i in range(N))
    target_net.eval()

    # Initialize replay memory
    memory = [ReplayMemory(REPLAY_SIZE) for _ in range(N)]

    @torch.no_grad
    def select_action(state, steps_done, individual):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                ac = policy_net[individual](state).max(1).indices
                ac = ac.view(1, 1)
                functional.reset_net(policy_net[individual])
                return ac
        else:
            return torch.tensor([[random.randrange(env[individual].action_space.n)]], device=device, dtype=torch.long)

    @torch.no_grad
    def get_loss(individual):
        if len(memory[individual]) < BATCH_SIZE:
            return

        transitions = memory[individual].sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net[individual](state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        functional.reset_net(target_net)

        # Compute the expected Q-Value
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Get the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        functional.reset_net(policy_net[individual])
        return loss
    
    
    @torch.no_grad
    def update_weights(loss):
        # Get a vector of all weights in the SNN
        lengths = [[policy_net[individual].fc[2*i].weight.flatten() for i in range(HIDDEN_LAYERS+2)] for individual in range(N)]
        weights = [torch.cat(length) for length in lengths]
        dimension = weights[0].shape

        if weights[0][0] not in [0,1]:
            weights = [torch.randint(0, 2, dimension).float().to(device) for _ in weights]

        change = [False for _ in range(N)]

        new_weights = []
        
        for individual in range(N):
            better = []
            if loss[individual] is not None:
                for n in NEIGHBOURS[individual]:
                    if loss[n] is not None and loss[n] < loss[individual]:
                        better.append(n)
            if better != []:
                mate = np.random.choice(better)
                bbefore = bit_flips(weights[mate], weights[individual])
                new_weights.append(combine_tensors(weights[mate], weights[individual]))
                bafter = bit_flips(weights[mate], new_weights[individual])
                # print(f"Individual {individual} is combining with individual {mate} because {round(float(loss[mate]), 3)} < {round(float(loss[individual]), 3)}\n They have copied {bbefore - bafter}/{weights[0].shape[0]} bits")
                change[individual] = True
            elif np.random.random() < 0.5:
                # print(f"Individual {individual} is mutating")
                new_weights.append(mutate_tensor(weights[individual], int(weights[individual].shape[0]*GAMMA+1)).float())
                change[individual] = True
            else:
                new_weights.append(weights[individual])
            
        for individual in range(N):
        # Update the weights to the new computed values if a change have been made
            if change[individual]:
                lengths[individual] = [i.size(dim=0) for i in lengths[individual]]
                cursor = 0
                for i in range(HIDDEN_LAYERS+2):
                    shape = policy_net[individual].fc[2*i].weight.size()
                    length = lengths[individual][i]
                    policy_net[individual].fc[2*i].weight = torch.nn.Parameter(data=new_weights[individual][cursor:(cursor+length)].view(shape), requires_grad=False)
                    cursor += length

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    max_reward = 0
    max_pt_path = os.path.join(model_dir, f'policy_net_{hidden_size}_max.pt')
    pt_path = os.path.join(model_dir, f'policy_net_{hidden_size}.pt')

    for i_episode in range(num_episodes):

        # Initialize the environment and state
        [env.reset() for env in env]
        state = [torch.zeros(size=[1, n_states], dtype=torch.float, device=device) for _ in range(N)]
        steps_done = 0
        total_reward = [0 for _ in range(N)]
        loss = [10 for _ in range(N)]
        done = [False for _ in range(N)]

        ended = 0

        # Repeat while there is less than 10 SNN which succeeded the task
        while ended < N-1:
            steps_done += 1
            for individual in range(N):

                if not done[individual]:
                    # Choose the best action given current policy_net
                    action = select_action(state[individual], steps_done, individual)
                    
                    # Get the next state and reward
                    next_state, reward, termination, truncation,_ = env[individual].step(action.item())
                    done[individual] = termination or truncation
                    total_reward[individual] += reward
                    next_state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)
                    reward = torch.tensor([reward], device=device)

                    if done[individual]:
                        next_state = None
                        ended += 1

                    # Save the current state in replay memory
                    memory[individual].push(state[individual], action, next_state, reward)

                    state[individual] = next_state
                    if done[individual] and total_reward[individual] > max_reward:
                        max_reward = total_reward[individual]
                        torch.save(policy_net[individual].state_dict(), max_pt_path)
                    

                    # Compute the loss function
                    loss[individual] = get_loss(individual)

                
            # Choose better policy nets
            update_weights(loss)

        print(f'Episode: {i_episode} Sum of rewards: {sum(total_reward)}')
        print(f'Total rewards are: {total_reward}')

        if i_episode % TARGET_UPDATE == 0:
            # print("Target:", target_net.fc[0].weight[:3])
            # print("Policy:", policy_net[0].fc[0].weight[:3])
            print("New target is:", np.argmax(total_reward))
            target_net.load_state_dict(torch.load(max_pt_path)) # policy_net[np.argmax(total_reward)].state_dict())

    print('complete')
    torch.save(policy_net.state_dict(), pt_path)
    print('state_dict path is', pt_path)

game = "CartPole-v1"

train(use_cuda=True, model_dir=f'./model/{game}', log_dir='./log', env_name=game, \
        hidden_size=128, num_episodes=50, seed=1)