# Q-Learning in Gym's "Taxi-v3" Environment

This repository contains a script that implements a reinforcement learning agent using the Q-learning algorithm in the Gym "Taxi-v3" environment. The "Taxi-v3" environment is a reinforcement learning scenario where a taxi must pick up and drop off passengers at specific locations within a grid. The objective of the agent is to learn an optimal policy to complete this task efficiently.

## Algorithm Description

Q-learning is a reinforcement learning method based on a table of values (Q-table), where each row represents a state of the environment and each column represents a possible action. The value in each cell indicates the expected value of taking a specific action in a given state.

In this script, the agent starts with a Q-table initialised with zeros and learns through interaction with the environment. During training, the agent receives rewards and adjusts the values in the Q table to reflect the best actions for each state, using the following update formula:

\[Q(s, a) = Q(s, a) + \alpha \times (R + \gamma \max_{a'}) Q(s', a') - Q(s, a))}]

where:
- \(s) is the current state,
- \(a) is the action taken,
- is the reward obtained,
- \(s'\) is the next state,
- \(a'\) is the possible actions of the next state,
- \(\alpha) is the learning rate, and
- \(\(\gamma) is the discount factor.

## Code structure

1. **Parameters and Configuration**:
   - The "Taxi-v3" environment is configured using `gym.make`.
   - Model parameters are defined: `alpha` (learning rate), `gamma` (discount factor), and `epsilon` (exploration probability).

2. **Training:
   - The agent trains for 1000 episodes, taking actions, earning rewards and updating the Q-table.
   - The agent explores by taking random actions with an `epsilon` probability and exploits the knowledge by taking the action with the highest value in the Q-table.
   - The rewards obtained are collected to visualise the progress of the training.

3. **Evaluation**:
   - After training, the agent is evaluated during 100 test episodes.
   - During the evaluation, the agent takes the action with the highest value in the Q-table, exploiting the learned policy.
   - The number of successfully completed episodes is calculated.

4. **Visualisation**:
   - The environment is visualised with `env.render()` to observe the behaviour of the taxi after training.
   - A graph is used to show the evolution of the rewards throughout the training.
   - ![](taxi_example.gif)

5. **Close the environment**:
   - The environment is closed with `env.close()` at the end of the script to free resources.

## How to Use the Code

1. Clone the repository on your local machine.
2. Make sure you have the necessary requirements installed.
3. Run the script to train the agent and visualise its behaviour in the "Taxi-v3" environment.
4. Observe the reward graph and the simulation of the environment to evaluate the agent's performance.
