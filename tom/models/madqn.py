import os 

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from torch import nn
import torch.optim as optim

from tqdm import tqdm
import copy

from argparse import ArgumentParser
from tom.models.common.networks import Policy
from tom.models.common.utils import ReplayBuffer
from tom.models.common.tree_search import best_first_search


class MADQN():

    def __init__(self, env, args):
        env.reset()
        self.env = env
        self.args = args 
        self.agents = env.agents
        self.num_agents = len(env.agents)
        # This self-play algorithm assumes all agents have the same folowing values:
        self.observation_space = self.env.observation_spaces[env.agents[0]]['observation']
        self.action_mask = self.env.observation_spaces[env.agents[0]]['action_mask']
        self.action_space = self.env.action_spaces[env.agents[0]]

        self.observation_shape = self.observation_space.shape
        self.action_shape      = self.action_space.shape

        self.observation_size = np.prod(self.observation_shape)
        self.action_size      = self.action_space.n
        
        self.device = args.device
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.layers = args.layers
        self.model_type = args.model_type
        self.e_min = 0.05
        self.e = 1 # start at 100% exploration 
        self.training_step = 0
        self.results = pandas.DataFrame()

        self.prev_mask = [None for _ in env.agents]
        self.prev_obs  = [None for _ in env.agents]
        self.prev_action  = [None for _ in env.agents]
        self.prev_rew  = [None for _ in env.agents]
        self.losses = []
        

        self.replay_buffer = ReplayBuffer(  buffer_size=int(1e5), 
                                            observation_shape=self.observation_size, 
                                            action_shape=(self.action_size,),
                                            belief_shape=())
        
        self.policy_net = Policy(input_size=self.observation_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.target_net = Policy(input_size=self.observation_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        
        if(os.path.exists(args.base_model + 'policy')): 
            print("loading base policy")
            self.policy_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.target_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.policy_net.eval()
            self.target_net.eval()


        self.all_parameters = list(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.all_parameters, lr=args.learning_rate)
    
    def reward_performance(self):
        env = copy.deepcopy(self.env)
        env.reset()

        cum_rewards = []
        eps_rewards = np.zeros(len(self.env.agents))

        for _ in range(10000):
            player_key = env.agent_selection
            player_index = env.agents.index(env.agent_selection)

            obs = env.observe(agent=player_key)['observation'].flatten().astype(np.float32)
            mask = env.observe(agent=player_key)['action_mask']

            if(player_index == self.args.player_id):
                action = self.predict(obs=obs, mask=mask)
            else:
                actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])

            env.step(action) 
            rew_n, done_n = list(env.rewards.values()), list(env.dones.values())
            eps_rewards += rew_n

            if all(done_n):
                env.reset()
                cum_rewards.append(eps_rewards)
                eps_rewards = np.zeros(len(self.env.agents))
                
        agent_reward = np.mean(cum_rewards, axis=0)[self.args.player_id]
        
        return agent_reward 

    def learn(self, total_timesteps):
        env = self.env
        env.reset()

        for learning_timestep in tqdm(range(total_timesteps + 1)):
            if(learning_timestep % self.args.log_time == 0 and not self.args.compare):
                log_tag = "models-" + str(int(learning_timestep / self.args.log_time))
                self.save(self.args.save_folder, log_tag = log_tag)
                
                if len(self.losses) > 0:
                    total_loss = np.mean(self.losses) 
                    reward_performance = self.reward_performance()
                    # not done correctly 
                    d = {"loss": total_loss, "performance": reward_performance, "learning_timestep":learning_timestep}
                    d = pandas.Series(data=d, index=['loss', 'performance', 'learning_timestep'])
                    print(d)
                    self.results = pandas.concat([self.results, d], ignore_index=True)
                    self.losses = []

            player_index = self.env.agents.index(env.agent_selection)
            obs = env.observe(agent=env.agent_selection)['observation'].flatten().astype(np.float32)
            mask = env.observe(agent=env.agent_selection)['action_mask']

            action = self.predict(obs=obs, mask=mask)
            env.step(action) 

            rew_n, done_n, info_n = list(env.rewards.values()), list(env.dones.values()), list(env.infos.values())
            obs_n = [env.observe(agent=agent_id)['observation'].flatten().astype(np.float32) for agent_id in self.env.agents]

            if(self.prev_rew[player_index] is not None):
                self.observe(self.prev_obs[player_index], obs, self.prev_action[player_index], self.prev_mask[player_index], mask, self.prev_rew[player_index], done_n[player_index], info_n[player_index])

            self.prev_obs[player_index]    = obs
            self.prev_mask[player_index]   = mask
            self.prev_action[player_index] = action 
            self.prev_rew[player_index]    = rew_n[player_index]

            self.on_step()

            if all(done_n):
                for agent_index in range(self.num_agents):
                    mask = env.observe(agent=env.agents[agent_index])['action_mask']
                    self.observe(self.prev_obs[agent_index], obs_n[agent_index], self.prev_action[agent_index], self.prev_mask[agent_index], mask, rew_n[agent_index], done_n[agent_index], info_n[agent_index])

                env.reset()
                self.prev_mask = [None for _ in env.agents]
                self.prev_obs  = [None for _ in env.agents]
                self.prev_action  = [None for _ in env.agents]
                self.prev_rew  = [None for _ in env.agents]

        return self.results

                
    def observe(self, obs, obs_next, action, mask, mask_next, reward, done, info):
        if(obs is None): return  
        self.replay_buffer.add(obs, obs_next, action, mask, mask_next, reward, done, info) 

    def on_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        self.training_step += 1
        (obs, next_obs, acts, masks, next_masks, dones, rewards) = self.replay_buffer.sample(self.batch_size, self.env)
        dones = dones.squeeze()
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = th.tensor(tuple(map(lambda s: s != 1, dones)), device=self.device, dtype=th.bool) 
        non_final_next_obs = np.asarray([s for index, s in enumerate(next_obs.cpu().numpy()) if dones[index] != 1])
        non_final_next_obs = th.from_numpy(non_final_next_obs).to(device=self.device) 
        rewards = rewards.squeeze() 

        non_final_next_masks = np.asarray([s for index, s in enumerate(next_masks.cpu().numpy()) if dones[index] != 1])
        non_final_next_masks = th.from_numpy(non_final_next_masks).to(device=self.device) 

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        policy_in = th.cat((obs.float(), masks.float()), 1)
        state_action_values = self.policy_net(policy_in).gather(dim=1, index=acts).squeeze()
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = th.zeros(self.batch_size, device=self.device)
        non_final_size = non_final_next_obs.size()[0]
        if(non_final_size > 0):
            # max besides zeroed out values 
            target_in = th.cat((non_final_next_obs.float(), non_final_next_masks.float()), 1)
            masked_next_state_values = self.target_net(target_in) 
            masked_next_state_indicies = masked_next_state_values - masked_next_state_values.min(1, keepdim=True)[0]
            masked_next_state_indicies /= masked_next_state_indicies.max(1, keepdim=True)[0] # normalize to between 0 and 1 
            masked_next_state_indicies *= non_final_next_masks
            indicies = masked_next_state_indicies.max(1)[1].detach()
            next_state_values[non_final_mask] = masked_next_state_values.gather(1, indicies.view(-1,1)).view(-1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards 

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.all_parameters:
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.losses.append(loss.cpu().detach().numpy())

        if self.training_step % self.target_update == 0: # Set target net to policy net
            self.target_net.load_state_dict(self.policy_net.state_dict()) 

    def predict(self, obs, mask):
        sample = np.random.random_sample()
        self.e *= 0.99 if self.e > self.e_min else self.e_min
        if(sample < self.e):
            actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
            action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
        else:
            with th.no_grad():
                obs_tensor = th.from_numpy(obs).to(self.args.device)
                mask_tensor = th.from_numpy(mask).to(self.args.device)
                policy_in = th.cat((obs_tensor, mask_tensor))
                q_values = self.policy_net(policy_in).cpu().numpy() 
                masked_q_values = [q for index, q in enumerate(q_values) if mask[index] == 1]
                action = np.where(q_values == np.max(masked_q_values))[0][0]
        
        return action

     
    def save(self, folder, log_tag):
        import os
        if not os.path.exists(folder + log_tag):
            os.makedirs(folder + log_tag)
        
        th.save(self.policy_net.state_dict(), folder + log_tag + "/policy")
    

    def compare(self, args):
        # load_folder=args.load_folder, opponent=args.opponent
        self.policy_net = Policy(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.policy_net.load_state_dict(th.load(args.load_folder + 'policy'))
        self.policy_net.eval()
        self.e = 0 

        if(self.args.opponent != "random"):
            opponent = Policy(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
            opponent.load_state_dict(th.load(args.opponent + 'policy'))
            opponent.eval()

        env = self.env
        env.reset()

        cum_rewards = []
        eps_rewards = np.zeros(len(self.env.agents))

        for learning_timestep in tqdm(range(args.timesteps)):
            player_key = env.agent_selection
            player_index = env.agents.index(env.agent_selection)

            obs = env.observe(agent=player_key)['observation'].flatten().astype(np.float32)
            mask = env.observe(agent=player_key)['action_mask']

            if(player_index == args.player_id):
                action = self.predict(obs=obs, mask=mask)
            else:
                if(self.args.opponent == "random"):
                    actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                    action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
                else:
                    with th.no_grad():
                        q_values = self.policy_net(th.from_numpy(obs).to(self.args.device)).cpu().numpy() 
                        masked_q_values = [q for index, q in enumerate(q_values) if mask[index] == 1]
                        action = np.where(q_values == np.max(masked_q_values))[0][0]
            
            env.step(action) 
            rew_n, done_n = list(env.rewards.values()), list(env.dones.values())
            eps_rewards += rew_n

            if all(done_n):
                env.reset()
                cum_rewards.append(eps_rewards)
                eps_rewards = np.zeros(len(self.env.agents))
        
        return cum_rewards