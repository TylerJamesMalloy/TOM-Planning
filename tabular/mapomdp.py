from itertools import permutations
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.DataFrame()

for it in range(10):
    for env_size in [2,3,4,5]:
        np.random.seed(it + env_size)
        num_players = env_size
        game_objects = env_size 
        objects_states = env_size 
        state_size = game_objects ** objects_states
        action_size = env_size 
        reward_size = num_players * state_size * action_size 

        #rewards = np.random.uniform(low=-1, high=1, size=(reward_size))
        rewards = np.random.normal(loc=0, scale=1, size=(reward_size))
        rewards = rewards.reshape(num_players, state_size, action_size)
        q_table = np.zeros((num_players, objects_states, action_size))

        state_observation_map = np.array([p for p in itertools.product(range(objects_states), repeat=game_objects)])

        epsilon = 1
        epsilon_min = 0.001 
        epsilon_rate = 0.99
        alpha   = 0.5
        gamma = 0.99
        cum_reward = 0
        cum_regret = 0

        prev_obs = [None for _ in range(num_players)]
        prev_rew = [None for _ in range(num_players)]
        prev_act = [None for _ in range(num_players)]

        state = np.random.choice(state_size)
        current_player = np.random.choice(num_players)

        for ts in range(1000):
            rand = np.random.uniform(0,1,1)[0]
            obs = state_observation_map[state][current_player]
            if(rand < epsilon):
                action = np.random.choice(action_size)
            else:
                action = np.argmax(q_table[current_player, obs, :])

            if(action > int(action_size/2)):
                new_state = state + 1 if state < (state_size-1) else 0
            elif(action < int(action_size/2)):
                new_state = state - 1 if state > 0 else (state_size-1)
            else:
                new_state = state 

            regret = (np.max(rewards[current_player, obs, :]) - rewards[current_player, obs, action])
            cum_reward += rewards[current_player, obs, action] 
            cum_regret += regret 

            data = data.append({"Timestep":ts, "Cumulative Regret":cum_regret, "Cumulative Reward":cum_reward, "Environment Size": env_size}, ignore_index=True)

            reward = rewards[current_player, state, action]

            if(prev_obs[current_player] is not None):
                # replace np.max(q) with most recent reward?
                q_table[current_player, prev_obs[current_player], prev_act[current_player]] = q_table[current_player, prev_obs[current_player], prev_act[current_player]] + (alpha * (prev_rew[current_player] + (gamma * np.max(q_table[current_player, obs, :]) - q_table[current_player, prev_obs[current_player], prev_act[current_player]])))
            
            prev_obs[current_player] = obs
            prev_rew[current_player] = reward
            prev_act[current_player] = action

            state = new_state
            epsilon = epsilon * epsilon_rate if epsilon > epsilon_min else epsilon_min
            current_player = current_player + 1 if current_player < num_players-1 else 0 

            
#data.to_pickle("./mapomdp.pkl")
# save data 
print("done training")
p = sns.lineplot(data=data, x="Timestep", y="Cumulative Regret", hue="Environment Size")
p.set_xlabel("Timestep", fontsize = 16)
p.set_ylabel("Cumulative Regret", fontsize = 16)
p.set_title("Cumulative Regret by Environment Size (Obs Only)", fontsize = 20)
#sns.lineplot(data=data, x="Timestep", y="Cumulative Reward", label="reward")
plt.show()
# graph regret and reward