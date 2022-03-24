from itertools import permutations
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from tqdm import tqdm 

# object-oriented multi-agent partially observable markov decision process 

data = pd.DataFrame()

alpha = 0.5
gamma = 0.99

timesteps = 10000
game_step = 0
game_length = 10 

env_size = 2

for env_size in [2,3,4,5]:
    for it in range(10):
        np.random.seed(it)
        epsilon = 1
        epsilon_min = 0.001 
        epsilon_rate = 0.99

        game_objects = env_size
        objects_states = env_size
        num_players = env_size
        action_size = env_size 

        obs_size = objects_states
        state_size = game_objects ** objects_states
        reward_size = num_players * state_size 

        # state_observation_map: maps the global state to a specific player's observation 
        state_observation_map = np.array([p for p in itertools.product(range(objects_states), repeat=game_objects)])

        def get_state(hidden):
            for som_idx, som in enumerate(state_observation_map):
                if(np.array_equal(som, hidden)):
                    return som_idx

        rewards = np.random.normal(loc=0, scale=.1, size=(reward_size))
        rewards = rewards.reshape(num_players, state_size)
        q_table = np.zeros((num_players, obs_size, action_size))

        state = np.random.choice(state_size)
        current_player = np.random.choice(num_players)


        prev_obs = [None for _ in range(num_players)]
        prev_rew = [None for _ in range(num_players)]
        prev_act = [None for _ in range(num_players)]
        cum_regret = [0 for _ in range(num_players)]

        #for ts in range(timesteps):
        print(" env_size: ", env_size, " it ", it)
        for ts in tqdm(range(timesteps)):
            obs = state_observation_map[state][current_player]

            rand = np.random.uniform(0,1,1)[0]
            if(rand < epsilon):
                action = np.random.choice(action_size)
            else:
                action = np.argmax(q_table[current_player, obs, :])

            next_hidden = state_observation_map[state]
            next_hidden[current_player] = action
            next_state = get_state(next_hidden)
            
            reward = 0
            if(prev_obs[current_player] is not None):
                q_table[current_player, prev_obs[current_player], prev_act[current_player]] += (alpha * (prev_rew[current_player] + (gamma * np.max(q_table[current_player, obs, :])) - q_table[current_player, prev_obs[current_player], prev_act[current_player]]))

            prev_obs[current_player] = obs
            prev_rew[current_player] = reward
            prev_act[current_player] = action

            if(game_step >= game_length): 
                done = True 
                for player_index in range(num_players):
                    reward = rewards[player_index, state]
                    q_table[player_index, prev_obs[player_index], prev_act[player_index]] += (alpha * (reward - q_table[player_index, prev_obs[player_index], prev_act[player_index]]))
                
                    possible_reward = []
                    for act in range(action_size):
                        possible_state = state_observation_map[state]
                        possible_state[current_player] = act
                        possible_state = get_state(possible_state)
                        possible_reward.append(rewards[player_index, possible_state])
                    
                    regret = np.max(np.array(possible_reward)) - reward 
                    cum_regret[current_player] += regret 
                    data = data.append({"Timestep":ts, "Cumulative Regret":cum_regret[current_player], "Episode Reward":reward, "Environment Size": env_size}, ignore_index=True)
                
                prev_obs = [None for _ in range(num_players)]
                prev_rew = [None for _ in range(num_players)]
                prev_act = [None for _ in range(num_players)]
                state = np.random.choice(state_size)
                current_player = np.random.choice(num_players)
                game_step = 0
            else:
                game_step += 1
                state = next_state
                current_player += 1 
                if(current_player >= num_players): current_player = 0

data.Timestep = (data.Timestep / 100).round().astype(int) * 100

print(data)
data.to_pickle("./marl.pkl")
# save data 
print("done training")
sns.lineplot(data=data, x="Timestep", y="Cumulative Regret", hue="Environment Size")
plt.show()
sns.lineplot(data=data, x="Timestep", y="Episode Reward", hue="Environment Size")
plt.show()
# graph regret and reward
        