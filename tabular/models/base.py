import numpy as np 

class Agent():
    def __init__(self, env, args):
        self.num_players = args.num_players
        self.num_objects = args.num_objects
        self.obj_states = args.obj_states
        self.num_actions = args.num_actions

        self.board_objects = (self.num_players + 1) * self.num_objects
        self.num_states = self.obj_states ** self.board_objects

        self.q_table = np.zeros(self.num_states, self.num_actions)

        self.env = env
    
    def predict(self, obs):
        q_values = self.q_table[obs,:]
        if(np.sum(q_values) == 0):
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(q_values)
        return action 

# Base RL 
class Base():
    def __init__(self, env, args):
        self.num_players = args.num_players
        self.num_objects = args.num_objects
        self.obj_states = args.obj_states

        self.env = env
        self.agents = [Agent(env, args) for _ in range(self.num_players)]
        
        self.current_player = env.get_player() 
    
    def learn(self, timesteps):
        ts = 0
        state = self.env.reset()
        while(ts < timesteps):
            ts += 1 
            action = self.agents[self.current_player].predict(self.state)
            done = self.env.step(action)

            new_state = self.env.get_state()
            reward = self.env.get_reward(self.current_player)

            self.agents[self.current_player].update(state, action, reward, new_state, done)
            
            
    
