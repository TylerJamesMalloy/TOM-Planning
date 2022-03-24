from itertools import permutations
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


# MARL MDP with game object like behaviour 

data = pd.DataFrame()

# add progress bar 
for it in range(10):
    np.random.seed(it)
    for env_size in [2,3,4,5]:
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
sns.lineplot(data=data, x="Timestep", y="Cumulative Regret", hue="Environment Size")
#sns.lineplot(data=data, x="Timestep", y="Cumulative Reward", label="reward")
plt.show()
# graph regret and reward

"""game_objects = 2
object_states = 2
num_players = 2
action_size = 2

gamma = 0.99 
lr = 0.1
epsilon = 0.1
epsilon_min = 0.05
epsilon_rate = 0.99

obs_size = game_objects * object_states
state_size = num_players ** obs_size
transitions_size = state_size * state_size  * num_players * action_size
reward_size = num_players * state_size  * action_size

data = pd.DataFrame()

# belief modelling 
# o-o attention modelling 
# tom modelling 

for _ in range(10):
    transitions = np.random.uniform(0,1,transitions_size)
    transitions = transitions.reshape(num_players, action_size, state_size, state_size)
    rewards = np.random.uniform(0,1,reward_size)
    rewards = rewards.reshape(num_players, state_size, action_size)

    state_observation_map = np.array([p for p in itertools.product(range(obs_size), repeat=num_players)])
    #state_observation_map = state_observation_map.reshape(state_size, num_players)

    q_table = np.zeros((num_players, obs_size, action_size))
    training_steps = 1000
    state = np.random.choice(state_size)
    current_player = np.random.choice(num_players)
    # make basic version RL agent for random learning game 
    # make a method for making random opponents based on human principles 
    # test basic version against random opponents 
    # make TOM-planning version of agent 
    # compare performance 

    # test with and without skip observations 
    prev_obs = [None for _ in range(num_players)]
    prev_rew = [None for _ in range(num_players)]
    cum_regret = 0 

    for ts in range(training_steps):
        obs = state_observation_map[state, current_player]  # get current player's observation

        rand = np.random.uniform(0,1,1)[0]
        if(rand < epsilon):
            action = np.random.choice(action_size)
        else:
            q_values = q_table[current_player, obs, :] # get current palyer's q-values for observation  
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
        if(prev_obs[current_player] is not None):
            q_table[current_player, prev_obs[current_player], action] += lr * (prev_rew[current_player] + gamma * np.max(q_table[current_player, obs, :] - q_table[current_player, prev_obs[current_player], action])) 

        prev_obs[current_player] = obs 
        prev_rew[current_player] = reward
        current_player = current_player + 1 if current_player < (num_players - 1) else 0

# save data 
print("done training")
sns.lineplot(data=data, x="Timestep", y="Cumulative Regret")
plt.show()
# graph regret and reward"""