import numpy as np
import torch as th 
import random 
import rlcard
import json 
import os 

from typing import Any, Dict, List

def get_belief_input(env, prev_action, prev_belief):
    if(str(env) == 'texas_holdem_v4' or str(env) ==  "texas_holdem_no_limit_v6" or str(env) ==  "leduc_holdem_v4"):
        if(prev_belief is None):
            if(str(env) == 'texas_holdem_v4' or str(env) ==  "texas_holdem_no_limit_v6" ):
                prev_belief = np.zeros(52)
            else:
                prev_belief = np.zeros(3)
        else:
            prev_belief = prev_belief.cpu().detach().numpy() 

        if(prev_action is None):
            if(str(env) == "texas_holdem_no_limit_v6"):
                prev_action = np.zeros(6)
            else:
                prev_action = np.zeros(4)
        else:
            if(str(env) == "texas_holdem_no_limit_v6"):
                prev_action_ohe = np.zeros(6)
            else:
                prev_action_ohe = np.zeros(4)
            prev_action_ohe[prev_action] = 1
            prev_action = prev_action_ohe

        public_cards = env.unwrapped.get_perfect_information()['public_card']
        if(public_cards is None):
            if(str(env) == 'texas_holdem_v4' or str(env) ==  "texas_holdem_no_limit_v6" ):
                public_cards = np.zeros(52)
            else:
                public_cards = np.zeros(3)
        else:
            if(str(env) == 'texas_holdem_v4' or str(env) ==  "texas_holdem_no_limit_v6" ):
                with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
                    card2index = json.load(file)
                    idx = [card2index[card] for card in public_cards]
                    public_cards = np.zeros(52)
                    public_cards[idx] = 1
            else:
                with open(os.path.join(rlcard.__path__[0], 'games/leducholdem/card2index.json'), 'r') as file:
                    card2index = json.load(file)
                    idx = card2index[public_cards] # one card 
                    public_cards = np.zeros(3)
                    public_cards[idx] = 1

        chips = np.array(env.unwrapped.get_perfect_information()['chips'])
        return  np.concatenate([prev_belief, prev_action, public_cards, chips])
    else:
        print("Error this environment is not implemented yet")
        assert(False)
    return 

class ReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple,
        action_shape: tuple,
        handle_timeout_termination: bool = True,

    ):
        self.n_envs = 1
        self.buffer_size = buffer_size
        self.obs_shape = observation_shape
        self.observations = np.zeros((self.buffer_size, self.obs_shape), dtype=np.int8)
        self.next_observations = np.zeros((self.buffer_size, self.obs_shape), dtype=np.int8)
        self.masks = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.int8)
        self.next_mask = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.int8)
        self.actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float16)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.int8)
        self.handle_timeout_termination = handle_timeout_termination
        self.pos = 0
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        mask:np.ndarray,
        next_mask:np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference 
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.masks[self.pos] = np.array(mask).copy()
        self.next_mask[self.pos] = np.array(next_mask).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def size(self):
        return self.pos 

    def sample(self, batch_size: int, env):
        #batch_inds = random.choices(np.linspace(0,self.buffer_size-1, num=self.buffer_size), k=batch_size)
        batch_inds = random.choices(np.linspace(0,self.pos-1, num=self.pos, dtype=np.int32), k=batch_size)
        data = (
            self.normalize_obs(self.observations[batch_inds, :], env),
            self.normalize_obs(self.next_observations[batch_inds, :], env),
            self.actions[batch_inds, 0, :],
            self.masks[batch_inds, 0, :],
            self.next_mask[batch_inds, 0, :],
            self.dones[batch_inds],
            self.normalize_reward(self.rewards[batch_inds], env),
        )
        return tuple(map(self.to_torch, data))
    
    def normalize_obs(self, obs, env):
        return obs 
    
    def normalize_reward(self, reward, env):
        return reward 
    
    def to_torch(self, input):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        return th.from_numpy(input).to(device)

class TOMReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple,
        action_shape: tuple,
        belief_shape: tuple,
        num_agents: int,
        handle_timeout_termination: bool = True,

    ):
        self.n_envs = 1
        self.buffer_size = buffer_size
        self.obs_shape = observation_shape
        self.observations = np.zeros((self.buffer_size, self.obs_shape), dtype=np.int32)
        self.next_observations = np.zeros((self.buffer_size, self.obs_shape), dtype=np.int32)
        self.masks = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.int32)
        self.next_mask = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.int32)
        self.prev_actions = np.zeros((self.buffer_size, self.n_envs, num_agents), dtype=np.int64)
        self.beliefs = np.zeros((self.buffer_size, self.n_envs, num_agents - 1) + belief_shape, np.int32) 
        self.prev_beliefs = np.zeros((self.buffer_size, self.n_envs, num_agents - 1) + belief_shape, np.int32) 
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.agent = np.zeros((self.buffer_size, self.n_envs, num_agents), np.int32) 
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        prev_action: np.ndarray,
        beliefs:np.ndarray,
        prev_belief: np.ndarray,
        mask:np.ndarray,
        next_mask:np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        agent: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference 
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.prev_actions[self.pos] = np.array(prev_action).copy() 
        self.beliefs[self.pos] = np.array(beliefs).copy()   
        self.prev_beliefs[self.pos] = np.array(prev_belief).copy()
        self.masks[self.pos] = np.array(mask).copy()
        self.next_mask[self.pos] = np.array(next_mask).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.agent[self.pos] = np.array(agent).copy()

        if self.handle_timeout_termination:
            #self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            self.timeouts[self.pos] = np.array([0])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def size(self):
        return self.pos 

    def sample(self, batch_size: int, env):
        #batch_inds = random.choices(np.linspace(0,self.buffer_size-1, num=self.buffer_size), k=batch_size)
        batch_inds = random.choices(np.linspace(0,self.pos-1, num=self.pos, dtype=np.int32), k=batch_size)
        data = (
            self.normalize_obs(self.observations[batch_inds, :], env),
            self.normalize_obs(self.next_observations[batch_inds, :], env),
            self.prev_actions[batch_inds, :],
            self.beliefs[batch_inds, :],
            self.prev_beliefs[batch_inds, :],
            self.masks[batch_inds, 0, :],
            self.next_mask[batch_inds, 0, :],
            self.normalize_reward(self.rewards[batch_inds], env),
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self.agent[batch_inds, 0, :],
        )
        return tuple(map(self.to_torch, data))
    
    def normalize_obs(self, obs, env):
        return obs 
    
    def normalize_reward(self, reward, env):
        return reward 
    
    def to_torch(self, input):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        return th.from_numpy(input).to(device)


"""
class TOMReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple,
        action_shape: tuple,
        belief_shape: tuple,
        num_agents: int,
        handle_timeout_termination: bool = True,

    ):
        self.n_envs = 1
        self.buffer_size = buffer_size
        self.obs_shape = observation_shape
        self.observations = np.zeros((self.buffer_size, self.obs_shape), dtype=np.int32)
        self.next_observations = np.zeros((self.buffer_size, self.obs_shape), dtype=np.int32)
        self.masks = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.int32)
        self.next_mask = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.int32)
        self.prev_actions = np.zeros((self.buffer_size, self.n_envs, num_agents), dtype=np.int64)
        self.beliefs = np.zeros((self.buffer_size, self.n_envs, num_agents - 1) + belief_shape, np.int32) 
        self.prev_beliefs = np.zeros((self.buffer_size, self.n_envs, num_agents - 1) + belief_shape, np.int32) 
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.agent = np.zeros((self.buffer_size, self.n_envs, num_agents), np.int32) 
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        prev_action: np.ndarray,
        beliefs:np.ndarray,
        prev_belief: np.ndarray,
        mask:np.ndarray,
        next_mask:np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        agent: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference 
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.prev_actions[self.pos] = np.array(prev_action).copy() 
        self.beliefs[self.pos] = np.array(beliefs).copy()   
        self.prev_beliefs[self.pos] = np.array(prev_belief).copy()
        self.masks[self.pos] = np.array(mask).copy()
        self.next_mask[self.pos] = np.array(next_mask).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.agent[self.pos] = np.array(agent).copy()

        if self.handle_timeout_termination:
            #self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            self.timeouts[self.pos] = np.array([0])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def size(self):
        return self.pos 

    def sample(self, batch_size: int, env):
        #batch_inds = random.choices(np.linspace(0,self.buffer_size-1, num=self.buffer_size), k=batch_size)
        batch_inds = random.choices(np.linspace(0,self.pos-1, num=self.pos, dtype=np.int32), k=batch_size)
        data = (
            self.normalize_obs(self.observations[batch_inds, :], env),
            self.normalize_obs(self.next_observations[batch_inds, :], env),
            self.prev_actions[batch_inds, :],
            self.beliefs[batch_inds, :],
            self.prev_beliefs[batch_inds, :],
            self.masks[batch_inds, 0, :],
            self.next_mask[batch_inds, 0, :],
            self.normalize_reward(self.rewards[batch_inds], env),
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self.agent[batch_inds, 0, :],
        )
        return tuple(map(self.to_torch, data))
    
    def normalize_obs(self, obs, env):
        return obs 
    
    def normalize_reward(self, reward, env):
        return reward 
    
    def to_torch(self, input):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        return th.from_numpy(input).to(device)
"""