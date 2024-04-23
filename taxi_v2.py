import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time

env = gym.make("Taxi-v3")

alpha = 0.4
gamma = 0.9
epsilon = 0.1

q_table = np.zeros((env.observation_space.n, env.action_space.n))

rewards = []

episodes = 1000

for episode in range(episodes):
    state_info = env.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info
    done = False
    total_reward = 0
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        step_info = env.step(action)
        next_state = step_info[0] if isinstance(step_info, tuple) else step_info
        reward = step_info[1]
        done = step_info[2]
        
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state
        total_reward += reward

    rewards.append(total_reward)

print("Entrenamiento completado.")
print("Simulación del taxi después del entrenamiento:")

test_episodes = 100
success_count = 0

for _ in range(test_episodes):
    state_info = env.reset()
    state = state_info[0] if isinstance(step_info, tuple)else state_info
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = np.argmax(q_table[state])
        
        step_info = env.step(action)
        next_state = step_info[0] if isinstance(step_info, tuple) else step_info
        done = step_info[2]
        
        state = next_state
        steps += 1
    
    if done and steps < 100:
        success_count += 1

print(f"El agente tuvo éxito en {success_count} de {test_episodes} episodios de prueba.")

env = gym.make("Taxi-v3", render_mode = "human")

tate_info = env.reset()
state = state_info[0] if isinstance(state_info, tuple) else state_info
done = False
steps = 0

while not done and steps < 100:
    action = np.argmax(q_table[state])
    
    step_info = env.step(action)
    next_state = step_info[0] if isinstance(step_info, tuple) else step_info
    done = step_info[2]
    
    env.render()
    time.sleep(0.5) 
    
    state = next_state
    steps += 1

plt.plot(range(episodes), rewards)
plt.title("Evolución de Recompensas en Taxi-v3")
plt.xlabel("Episodios")
plt.ylabel("Recompensas Acumuladas")
plt.show()

env.close()
