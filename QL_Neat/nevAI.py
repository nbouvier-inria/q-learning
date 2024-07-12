from collections import namedtuple
import numpy as np
from utility import ReplayMemory, Transition, NpEncoder
from spikingjelly.activation_based import functional
import neat_python as neat
import math
import gymnasium as gym
import random
from tqdm import tqdm


""" Hyperparameters """
# Memory and rewards
BATCH_SIZE = 128
REPLAY_SIZE = 10000
FINAL_REWARD = 0

# General simulation control
GENERATIONS = 5000
EPS_START = 1
EPS_END = 0.0001
EPS_DECAY = GENERATIONS/10

# Discount factor control
EPSILON = 0.1
GAMMA_START = 1 - EPSILON
GAMMA_END = 1 - EPSILON
GAMMA_DECAY = 50

# Neural network architecture
HIDDEN_LAYERS = 2
VTH = 1.
T = 16 

""" Parameters """
env_name = "CartPole-v1"
    
""" Learning scripts """
def q_learning(env_name):
    env = gym.make(env_name, render_mode="rgb_array").unwrapped

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    hp = neat.Hyperparameters()
    # hp.mutation_probabilities['node'] *= 2
    # hp.mutation_probabilities['edge'] *= 2

    memory = ReplayMemory(REPLAY_SIZE, reset=False)

    brain = neat.Brain(n_states, n_actions, population=BATCH_SIZE, hyperparams=hp)
    brain.generate()

    def select_action(state, steps_done, policy):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            ac = np.argmax(policy.forward(state))
            return ac
        else:
            return random.randrange(n_actions)

    def save():
        # try:
        #     brain.save("brain")
        # except NameError:
        #     pass
        memory.save()

    def get_loss(policy, time_step):

        if len(memory) < BATCH_SIZE:
            return 0
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # non_final_mask[state] if and only if state is non final
        non_final_mask = np.array(tuple(map(lambda s: s is not None,
                                                batch.next_state)))
        # List of non final states
        non_final_next_states = [s for s in batch.next_state
                                                    if s is not None]
        # Compute the Q-Value
        state_action_values = [policy.forward(state)[action] for state, action in zip(batch.state, batch.action)]

        # Compute maximum Q-Value of future states
        next_state_values = np.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = [max(policy.forward(state)) for state in non_final_next_states]

        # Compute the expected Q-Value
        gamma = (GAMMA_END + (GAMMA_START - GAMMA_END) * \
                        math.exp(-1. * time_step/ GAMMA_DECAY))
        expected_state_action_values = (next_state_values * gamma) + np.where(non_final_mask, batch.reward, FINAL_REWARD)

        # Get the loss
        loss = np.average([(value - expected)**2 for value, expected in zip(state_action_values, expected_state_action_values)])

        # Print informations about the loss
        # if time_step%100==0:
        #     print("Loss: ", round(100*loss,1))
        # print(state_action_values[:10], expected_state_action_values[:10])

        return 1000-100*loss

    time_steps = 0
    done = True
    i_episode = 0
    total_reward = 0
    try:
        for time_steps in tqdm(range(GENERATIONS)):
            if done:
                i_episode += 1
                try:
                    print(f"Reward is {total_reward}, epsilon is {round(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * time_steps / EPS_DECAY), 2)}, gamma is {round((GAMMA_END + (GAMMA_START - GAMMA_END) * math.exp(-1. * time_steps/ GAMMA_DECAY)), 3)} and current best has loss {round(-get_loss(best, time_steps), 1)}")
                    print([f"{edge}:{best._edges[edge].weight}" for edge in best._edges])
                    print(best._fitness)
                except NameError:
                    pass
                env.reset()
                state = np.zeros(n_states)
                total_reward = 0
            
            if brain.should_evolve():
                # Choose the best action given current policy_net
                best = brain.get_fittest()
                action = select_action(state, time_steps, best)

                # Get the next state and reward
                next_state, reward, termination, truncation,_ = env.step(action)
                done = termination or truncation
                total_reward += reward

                # Save the current state in replay memory
                current_state = Transition(state, action, next_state, reward)
                memory.push(state, action, next_state, reward)

                state = next_state
                brain.evaluate_parallel(get_loss, time_steps)
        save()
    except KeyboardInterrupt:
        save()


env = gym.make(env_name, render_mode="rgb_array").unwrapped
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

hp = neat.Hyperparameters()
# hp.mutation_probabilities['node'] *= 2
# hp.mutation_probabilities['edge'] *= 2

memory = ReplayMemory(REPLAY_SIZE, reset=False)

brain = neat.Brain(n_states, n_actions, population=BATCH_SIZE, hyperparams=hp)
brain.generate()

env.close()

def get_reward(policy, env_name):
    print("Generating")
    env = gym.make(env_name)
    print("Generated")
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy.forward(state)
        state, reward, termination, truncation, _ = env.step(action)
        done = termination or truncation
        total_reward += reward
    env.close()
    return total_reward

for time_steps in tqdm(range(GENERATIONS)):
    
    if brain.should_evolve():
        brain.evaluate_parallel(get_reward, time_steps)
    
    best = brain.get_fittest()
    print(best.get_fitness())

""" Execution """
# reinforcment(env_name=env_name)
