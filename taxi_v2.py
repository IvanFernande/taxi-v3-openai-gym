import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Set up the environment
env = gym.make("Taxi-v3")

# Model parameters
alpha = 0.4  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Collect rewards for visualization
rewards = []

# Number of training episodes
episodes = 1000

# Q-learning training loop
for episode in range(episodes):
    # Reset the environment and get the initial state
    state_info = env.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info  # Handle tuple for Gym's reset output
    done = False
    total_reward = 0  # Total reward for the current episode
    
    # Loop until the episode is done
    while not done:
        # Choose an action (explore or exploit)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Random action (explore)
        else:
            action = np.argmax(q_table[state])  # Best action from Q-table (exploit)
        
        # Perform the action and get the next state, reward, and done flag
        step_info = env.step(action)
        next_state = step_info[0] if isinstance(step_info, tuple) else step_info
        reward = step_info[1]
        done = step_info[2]
        
        # Update Q-table using the Bellman equation
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        # Update the current state
        state = next_state
        total_reward += reward  # Accumulate the reward

    # Store the total reward for this episode
    rewards.append(total_reward)

# Training is complete
print("Training completed.")
print("Simulating the taxi after training:")

# Testing and evaluation
test_episodes = 100
success_count = 0

for _ in range(test_episodes):
    # Reset the environment for each test
    state_info = env.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info
    done = False
    steps = 0
    
    # Loop until the episode is done or a step limit is reached
    while not done and steps < 100:
        action = np.argmax(q_table[state])  # Choose the best action
        
        # Perform the action
        step_info = env.step(action)
        next_state = step_info[0] if isinstance(step_info, tuple) else step_info
        done = step_info[2]
        
        # Update the state
        state = next_state
        steps += 1
    
    # If the episode completes successfully within 100 steps, increment the success count
    if done and steps < 100:
        success_count += 1

print(f"The agent succeeded in {success_count} of {test_episodes} test episodes.")

# Visualization with human render mode
env = gym.make("Taxi-v3", render_mode="human")

state_info = env.reset()
state = state_info[0] if isinstance(state_info, tuple) else state_info
done = False
steps = 0

# Render the environment after training to see the taxi's behavior
while not done and steps < 100:
    action = np.argmax(q_table[state])  # Choose the best action
    
    step_info = env.step(action)
    next_state = step_info[0] if isinstance(step_info, tuple) else step_info
    done = step_info[2]
    
    env.render()  # Visualize the environment after each action
    time.sleep(0.5)  # Pause to allow observation
    
    state = next_state
    steps += 1

# Plot the reward evolution during training
plt.plot(range(episodes), rewards)
plt.title("Reward Evolution in Taxi-v3")
plt.xlabel("Episodes")
plt.ylabel("Accumulated Rewards")
plt.show()

# Close the environment
env.close()
plt.ylabel("Recompensas Acumuladas")
plt.show()

env.close()
