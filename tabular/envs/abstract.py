import numpy as np 

class AbstractEnv():
    def __init__(self, args):
        self.num_players = args.num_players
        self.num_objects = args.num_objects # num actions equals number of observed objects
        self.obj_states = args.obj_states
        self.episode_length = args.episode_length

        self.board_objects = (self.num_players + 1) * self.num_objects
        self.board_shape   = ((self.num_players + 1) , self.num_objects)

        self.num_states = self.obj_states ** self.board_objects

        self.board = np.array([np.random.choice(self.obj_states) for _ in range(self.board_objects)]).reshape(self.board_shape)
        self.current_state = self.get_state()
        self.current_player = np.random.choice(self.num_players)

        self.changes = [np.random.uniform() for _ in range(self.num_objects)]
        self.rewards = [np.random.uniform() for _ in range(self.num_objects)]

        self.episode_step = 0 
    
    def reset(self):
        self.board = np.array([np.random.choice(self.obj_states) for _ in range(self.board_objects)]).reshape(self.board_shape)
        self.current_state = self.get_state()
        self.current_player = np.random.choice(self.num_players)
        self.episode_step = 0 

        return self.current_state
    
    def step(self, action):
        done = 0

        # first index is shared objects, then non-shared
        if(np.random.uniform() < self.changes[action]):
            self.board[0,action] = self.board[self.current_player+1,action]
        self.episode_step += 1 
        if(self.episode_step >= self.episode_length):
            self.reset()
            done = 1 

        self.current_player = self.current_player + 1 if self.current_player < self.num_players else 0
        return done 

    def get_reward(self, player):
         return 0 
        
    def get_obs(self):
        shared = self.board[0,:]
        hand = self.board[self.current_player+1,:]
        return np.concatenate((shared,hand))

    def get_state(self):
        return sum(obj * (self.obj_states ** obj_idx) for obj_idx, obj in enumerate(self.board.flatten()))
    
    def get_player(self):
        return self.current_player