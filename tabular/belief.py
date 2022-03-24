from itertools import permutations
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import copy 

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

        obs_size = objects_states
        opp_actions = action_size ** (num_players-1) 
        #rewards = np.random.uniform(low=-1, high=1, size=(reward_size))
        rewards = np.random.normal(loc=0, scale=1, size=(reward_size))
        rewards = rewards.reshape(num_players, state_size, action_size)
        q_table = np.zeros((num_players, state_size, action_size))
        b_table = np.zeros((num_players, obs_size, opp_actions, state_size))

        state_observation_map = np.array([p for p in itertools.product(range(objects_states), repeat=game_objects)])
        opponent_action_map   = np.array([p for p in itertools.product(range(action_size), repeat=num_players-1)]) 

        epsilon = 1
        epsilon_min = 0.001 
        epsilon_rate = 0.99
        alpha   = 0.8
        gamma = 0.99
        cum_reward = 0
        cum_regret = 0

        prev_obs = [None for _ in range(num_players)]
        prev_state_pred = [None for _ in range(num_players)]
        prev_state = [None for _ in range(num_players)]
        prev_rew = [None for _ in range(num_players)]
        prev_act = [None for _ in range(num_players)]
        prev_opp_act = [None for _ in range(num_players)]

        state = np.random.choice(state_size)
        current_player = np.random.choice(num_players)

        for ts in range(1000):
            rand = np.random.uniform(0,1,1)[0]
            obs = state_observation_map[state][current_player]
            opp_act = 0
            if(not any(elem is None for elem in prev_act)):
                opponent_actions = copy.copy(prev_act)
                opponent_actions.pop(current_player) 
                for map_index, map_act in enumerate(opponent_action_map):
                    if(np.array_equal(map_act,opponent_actions)):
                        opp_act = map_index
                state_pred = np.argmax(b_table[current_player, obs, opp_act, :])
            else:
                state_pred = np.random.choice(state_size)

            #print("predicted state is: ", state_pred, " real state is: ", state, " state size is: ", state_size)
            state_pred_accuracy = 1 if state_pred == state else 0 

            if(rand < epsilon):
                action = np.random.choice(action_size)
            else:
                action = np.argmax(q_table[current_player, state_pred, :])

            if(action > int(action_size/2)):
                new_state = state + 1 if state < (state_size-1) else 0
            elif(action < int(action_size/2)):
                new_state = state - 1 if state > 0 else (state_size-1)
            else:
                new_state = state 

            regret = (np.max(rewards[current_player, obs, :]) - rewards[current_player, obs, action])
            cum_reward += rewards[current_player, obs, action] 
            cum_regret += regret 

            data = data.append({"Timestep":ts, "Cumulative Regret":cum_regret, "Cumulative Reward":cum_reward, "Environment Size": env_size, "Belief Accuracy":state_pred_accuracy}, ignore_index=True)

            reward = rewards[current_player, state, action]

            if(prev_state_pred[current_player] is not None):
                # replace np.max(q) with most recent reward?
                prior = q_table[current_player, prev_state_pred[current_player], prev_act[current_player]]
                q_table[current_player, prev_state_pred[current_player], prev_act[current_player]] = prior + (alpha * (prev_rew[current_player] + (gamma * np.max(q_table[current_player, state_pred, :]) - prior)))
                
                # try some bayesian method for belief prediction update? 
                if(prev_opp_act[current_player] is not None):
                    for possible_state in range(state_size):
                        if(np.array_equal(possible_state, state)):
                            b_table[current_player, prev_obs[current_player], prev_opp_act[current_player], state] += 1
                        else:
                            b_table[current_player, prev_obs[current_player], prev_opp_act[current_player], state] -= 1

            prev_obs[current_player] = obs
            prev_rew[current_player] = reward
            prev_act[current_player] = action
            prev_state_pred[current_player] = state_pred
            prev_state[current_player] = state 
            prev_opp_act[current_player] = opp_act

            state = new_state
            epsilon = epsilon * epsilon_rate if epsilon > epsilon_min else epsilon_min
            current_player = current_player + 1 if current_player < num_players-1 else 0 


data.Timestep = (data.Timestep / 10).round().astype(int) * 10
#data.to_pickle("./mapomdp.pkl")
# save data 
print("done training")
p = sns.lineplot(data=data, x="Timestep", y="Belief Accuracy", hue="Environment Size")
p.set_xlabel("Timestep", fontsize = 16)
p.set_ylabel("State Predictio Accuracy", fontsize = 16)
p.set_title("State Prediction Accuracy by Environment Size", fontsize = 20)
#sns.lineplot(data=data, x="Timestep", y="Cumulative Reward", label="reward")
plt.show()
# graph regret and reward