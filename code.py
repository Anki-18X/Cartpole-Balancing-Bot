import gym
import numpy as np
import random

# Set up the CartPole environment
env = gym.make('CartPole-v1')

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 1.0    # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration probability
episodes = 1000   # Number of training episodes
max_steps = 200   # Max steps per episode

# Discretization settings
n_buckets = (6, 6, 12, 12)  # Number of buckets per state dimension
q_table = np.zeros(n_buckets + (env.action_space.n,))

# Discretize the state space
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-np.radians(50), np.radians(50)]

def discretize_state(state):
    state_adj = (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
    discretized = [int(np.digitize(state_adj[i], np.linspace(0, 1, n_buckets[i])) - 1) for i in range(len(state))]
    return tuple(discretized)

# Training loop
for episode in range(episodes):
    current_state = discretize_state(env.reset())
    done = False
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])

        # Perform action
        obs, reward, done, _ = env.step(action)
        new_state = discretize_state(obs)

        # Update Q-table
        best_future_q = np.max(q_table[new_state])
        q_table[current_state + (action,)] += alpha * (reward + gamma * best_future_q - q_table[current_state + (action,)])

        current_state = new_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Output progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
