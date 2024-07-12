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
from random import choice
import torch

def voltage_modify(x, T):
        """
        Modification of x to widen or recenter the output
        """
        # x = 2*x/T
        return x #torch.where(x > 1, x - 1, 1-1/x)

class DQSN(nn.Module):
    """ Spiking neural network 
    Args:
        input_size: input size of network
        hidden_size: hidden size of network
        output_size: output size of network
        T: time steps to run the neural network for
        Vth: Voltage threshold
        hidden_layers: number of hidden layers
        simplify: mask for the entry layer
        normalisation: mask for the output layer
    """
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=1, T=16, Vth=1., simplify=lambda x:x, normalisation=voltage_modify):
        super().__init__()

        self.simplify = simplify
        self.voltage_modify = normalisation
        self.hidden_layers = hidden_layers
        layers = OrderedDict()
        layers["0th_layer"] = layer.Linear(input_size, hidden_size)
        layers["0th_IF"] = neuron.IFNode(v_threshold=Vth)
        for i in range(hidden_layers):
            layers[f"{i+1}th_laye"] = layer.Linear(hidden_size, hidden_size)
            layers[f"{i+1}th_IF"] = neuron.IFNode(v_threshold=Vth)
        layers[f"{hidden_layers+1}th_layer"] = layer.Linear(hidden_size, output_size)
        layers[f"output_layer"] = NonSpikingLIFNode(tau=2.0)

        self.fc = nn.Sequential(
            layers
        )

        self.T = T

    def forward(self, x):
        entry = self.simplify(x)
        for t in range(self.T):
            self.fc(entry)
        return self.voltage_modify(self.fc[-1].v, self.T)

    def to_tensor(self):
        return torch.cat([self.fc[2*i].weight.flatten() for i in range(self.hidden_layers+2)])
    
    def from_tensor(self, weights):
        lengths = [self.fc[2*i].weight.flatten().size(dim=0) for i in range(self.hidden_layers+2)]
        cursor = 0
        for i in range(len(lengths)):
            shape = self.fc[2*i].weight.size()
            self.fc[2*i].weight = torch.nn.Parameter(data=weights[cursor:(cursor+lengths[i])].view(shape), requires_grad=False)
            cursor += lengths[i]

def torus_no_comm(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Returns a torus graph with n nodes and no communication edges.
    """
    n = int(np.sqrt(n))
    V = [i for i in range(n**2)]
    E = []
    return V, E

def torus(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Square Torus-shaped graph (V, E) of
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


def combine_tensors_1point(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Combine two tensors of the same shape, where a sequence
    of a is chosen to replace thecorresponding bits of b.
    Well suited for neural networks weights crossovers
    Args:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor
    Returns:
        torch.Tensor: Combined tensor
    """

    # Get the lengths of the tensors
    len1 = a.shape[0]
    len2 = b.shape[0]

    # Choose a random number of consecutive elements to take from tensor1
    take1 = torch.randint(0, len1, (1,)).item()
    take2 = torch.randint(0, len2, (1,)).item()
    temp = take1
    take1 = min(take1, take2)
    take2 = max(temp, take2)

    # Take the random number of consecutive elements from tensor1
    take_from_tensor1 =a[:take1]
    take_from_tensor2 = b[take1:take2]
    take_from_tensor3 =a[take2:]

    # Combine the two tensors
    combined_tensor = torch.cat((take_from_tensor1, take_from_tensor2, take_from_tensor3))

    return combined_tensor


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



def mutate_tensor(tensor, k):
    """
    Mutate a binary tensor by flipping k random bits.
    Args:
        tensor (torch.Tensor): Binary tensor (-1s, 0s and 1s)
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
        """ Change those lines to choose precision """
        value = torch.randint(-1, 2, (1,)).item()
        # value = 2*torch.randint(0,2, (1,)).item() - 1
        result.view(-1)[idx] = value
    return result


class NonSpikingLIFNode(neuron.LIFNode):
    """" Litterally just a non spiking
    LIF Node. Thou might call it a LI if 
    thou like """
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
