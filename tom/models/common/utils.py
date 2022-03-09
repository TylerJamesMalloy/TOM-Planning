import numpy as np
import torch as th 
import random 

from typing import Any, Dict, List


class ReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple,
        action_shape: tuple,
        belief_shape: tuple,
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
        self.actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.int64)
        self.prev_actions = np.zeros((self.buffer_size, self.n_envs, num_agents), dtype=np.int64)
        self.prev_beliefs = np.zeros((self.buffer_size, self.n_envs) + belief_shape, np.int32) 
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        prev_belief: np.ndarray,
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
        self.prev_actions[self.pos] = np.array(prev_action).copy()    
        self.prev_beliefs[self.pos] = np.array(prev_belief).copy()
        self.masks[self.pos] = np.array(mask).copy()
        self.next_mask[self.pos] = np.array(next_mask).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

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
            self.actions[batch_inds, 0, :],
            self.prev_actions[batch_inds, 0, :],
            self.prev_beliefs[batch_inds, :],
            self.masks[batch_inds, 0, :],
            self.next_mask[batch_inds, 0, :],
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
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