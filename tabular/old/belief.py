from itertools import permutations
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

game_objects = 3
object_states = 3
num_players = 3
action_size = 3

gamma = 0.99 
lr = 0.1
epsilon = 0.1
epsilon_min = 0.05
epsilon_rate = 0.99

obs_size = game_objects * object_states
state_size = num_players * obs_size
transitions_size = state_size * state_size  * num_players * action_size
reward_size = num_players * state_size  * action_size

data = pd.DataFrame()
for _ in range(1):
    transitions = np.random.uniform(0,1,transitions_size)
    transitions = transitions.reshape(num_players, action_size, state_size, state_size)
    rewards = np.random.uniform(0,1,reward_size)
    rewards = rewards.reshape(num_players, state_size, action_size)

    state_observation_map = np.array([p for p in itertools.product(range(obs_size), repeat=num_players)])
    #state_observation_map = state_observation_map.reshape(state_size, num_players)
    opponent_action_size = ((num_players-1) ** (action_size+1))
    prev_actions_map = np.array([range(opponent_action_size)])
    prev_actions_shape = tuple([action_size+1 for _ in range(num_players-1)])
    prev_actions_map = prev_actions_map.reshape(prev_actions_shape)
    b_table = np.zeros((obs_size+1, (opponent_action_size), state_size))
    q_table = np.zeros((num_players, state_size, action_size))
    training_steps = 10000
    state = np.random.choice(state_size)
    current_player = np.random.choice(num_players)
    # belief modelling 
    

    # test with and without skip observations?
    prev_state_predictions = [None for _ in range(num_players)]
    prev_rews   = [None for _ in range(num_players)]
    prev_obss   = [0 for _ in range(num_players)]
    prev_acts   = [0 for _ in range(num_players)]
    cum_regret = 0 

    for ts in range(training_steps):
        obs = state_observation_map[state, current_player]  # get current player's observation
        opponent_previous_actions = [x for i,x in enumerate(prev_acts) if i!=current_player]
        mapped_prev_actions = prev_actions_map[tuple(opponent_previous_actions)]
        state_prediction = np.argmax(b_table[prev_obss[current_player], mapped_prev_actions, :])

        rand = np.random.uniform(0,1,1)[0]
        if(rand < epsilon):
            action = np.random.choice(action_size)
        else:
            #q_values = q_table[current_player, state_prediction, :] # get current palyer's q-values for observation  
            q_values = q_table[current_player, state, :] # get current palyer's q-values for observation 
            action = np.argmax(q_values)               # select action with highest q-value 
            #action = np.argmax(rewards[current_player, state, :]) # for testing, choose action omnisciently 
            cum_regret += (np.max(rewards[current_player, state, :]) - rewards[current_player, state, action])

        epsilon = epsilon * epsilon_rate if epsilon > epsilon_min else epsilon_min

        transition = transitions[current_player, action, state, :] # get state transition
        transition /= np.sum(transition) # normalize transition to probability 
        state = np.random.choice(state_size, p=transition) # get next state 
        reward = rewards[current_player, state, action] # get reward
        data = data.append({"Timestep":ts, "Cumulative Regret":cum_regret}, ignore_index=True)

        new_obs = state_observation_map[state, current_player]
        # Q(S,A) = Q(S,A) + a * (r + g * max_A(Q(S',A)) — Q(S,A))
        if(prev_rews[current_player] is not None):
            q_table[current_player, prev_state_predictions[current_player], action] += lr * (prev_rews[current_player] + gamma * np.max(q_table[current_player, state_prediction, :] - q_table[current_player, prev_state_predictions[current_player], action])) 
            # train b table using 'omniscient' knowledge of state 
            b_table[prev_obss[current_player], mapped_prev_actions, state] = 1

        prev_state_predictions[current_player] = state_prediction
        prev_rews[current_player] = reward
        prev_obss[current_player] = obs + 1
        prev_acts[current_player] = action + 1
        current_player = current_player + 1 if current_player < (num_players - 1) else 0

# save data 
print("done training")
sns.lineplot(data=data, x="Timestep", y="Cumulative Regret")
plt.show()
# graph regret and reward