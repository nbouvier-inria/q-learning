import json
from typing import Tuple, List
import numpy as np
from collections import namedtuple
import random

Transition: Tuple[np.ndarray, int, np.ndarray, int] = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class NpEncoder(json.JSONEncoder):
    """
    Encoder to normalize datas before
    saving them with json.dump
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Transition):
            return obj._asdict()
        if isinstance(obj, dict) and isinstance(obj[0], Tuple):
            return {k: super(NpEncoder, self).default(v) for k, v in obj}
        return super(NpEncoder, self).default(obj)

class ReplayMemory(object):
    def __init__(self, capacity, reset=False):
        """ If there is a memory.json file,
        ignore its arguments and load it instead
        If reset is set to true, the memory.json
        will be overwritten """
        try:
            if reset:
                raise Exception
            self.load()
        except: 
            self.capacity = capacity
            self.memory: list = []
            self.position = 0

    def set_capacity(self, capacity):
        """ Changes the memory capacity and truncates the
        memory if needed. Only truncates old datas, that is
        to say datas before self.position  """
        self.capacity = capacity
        if len(self.memory) > self.capacity:
            self.memory = self.memory[self.position:]
            self.position = max(0, self.position - self.capacity)
        if len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self):
        """ Save the current memory, position and capacity
        of the memory into a JSON file"""
        with open("memory.json", "w") as f:
            json.dump({"memory": self.memory, "position": self.position, "capacity": self.capacity
                       }, f, cls=NpEncoder)
            print("Memory saved")
    
    def load(self):
        """ Load the memory, position and capacity
        of the memory from a JSON file"""
        with open("memory.json", "r") as f:
            data = json.load(f)
            self.memory = [Transition(np.array(s), a, np.array(ns), r )for (s, a, ns, r)in data["memory"]]
            self.position = data["position"]
            self.capacity = data["capacity"]
            print("Memory loaded")

    def __len__(self):
        return len(self.memory)