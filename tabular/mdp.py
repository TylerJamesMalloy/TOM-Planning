from itertools import permutations
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.DataFrame()

for it in range(10):
    np.random.seed(it)
    for env_size in [2,3,4,5,6]:
        game_objects = env_size 
        objects_states = env_size 
        state_size = game_objects ** objects_states
        action_size = env_size 
        reward_size = state_size * action_size 

        #rewards = np.random.uniform(low=-1, high=1, size=(reward_size))
        rewards = np.random.normal(loc=0, scale=.1, size=(reward_size))
        rewards = rewards.reshape(state_size, action_size)
        q_table = np.zeros_like(rewards)

        epsilon = 1
        epsilon_min = 0.001 
        epsilon_rate = 0.99
        alpha   = 0.5
        gamma = 0.99
        cum_reward = 0
        cum_regret = 0

        state = np.random.choice(state_size)

        for ts in range(1000):
            rand = np.random.uniform(0,1,1)[0]
            if(rand < epsilon):
                action = np.random.choice(action_size)
            else:
                action = np.argmax(q_table[state, :])

            if(action > int(env_size/2)):
                new_state = state + 1 if state < env_size else 0
            elif(action < int(env_size/2)):
                new_state = state - 1 if state > 0 else env_size
            else:
                new_state = state 
            

            regret = (np.max(rewards[state, :]) - rewards[state, action])
            cum_reward += rewards[state, action] 
            cum_regret += regret 

            data = data.append({"Timestep":ts, "Cumulative Regret":cum_regret, "Cumulative Reward":cum_reward, "Environment Size": env_size}, ignore_index=True)

            reward = rewards[state, action]
            q_table[state, action] = q_table[state, action] + (alpha * (reward + (gamma * np.max(q_table[new_state, :]) - q_table[state, action])))
            state = new_state

            
            epsilon = epsilon * epsilon_rate if epsilon > epsilon_min else epsilon_min

            
data.to_pickle("./rl.pkl")
# save data 
print("done training")
p = sns.lineplot(data=data, x="Timestep", y="Cumulative Regret", hue="Environment Size")
p.set_xlabel("Timestep", fontsize = 18)
p.set_ylabel("Cumulative Regret", fontsize = 18)
p.set_title("Cumulative Regret by Environment Size", fontsize = 24)
#sns.lineplot(data=data, x="Timestep", y="Cumulative Reward", label="reward")
plt.show()
# graph regret and reward