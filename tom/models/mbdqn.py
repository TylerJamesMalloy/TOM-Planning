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
from tom.models.common.networks import Policy, Transition 
from tom.models.common.utils import ReplayBuffer
from tom.models.common.tree_search import best_first_search



class MBDQN():
    """
    Deep Model-Based Q-Network (MADQN)
    """

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
        
        self.planning = args.planning
        self.device = args.device
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.layers = args.layers
        self.model_type = args.model_type
        self.e_min = 0.05
        self.e = 1 # start at 100% exploration 
        self.training_step = 0
        self.loss_log = []

        self.prev_mask = [None for _ in env.agents]
        self.prev_obs  = [None for _ in env.agents]
        self.prev_action  = [None for _ in env.agents]
        self.prev_rew  = [None for _ in env.agents]
        

        self.replay_buffer = ReplayBuffer(  buffer_size=int(1e5), 
                                            observation_shape=self.observation_size, 
                                            action_shape=(self.action_size,),
                                            belief_shape=())
        
        self.policy_net = Policy(input_size=self.observation_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.target_net = Policy(input_size=self.observation_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.transition_net = Transition(state_size=self.observation_size, action_size=self.action_size, mask_size=self.action_size, layers=args.layers).to(self.args.device)
        print(args.base_model + 'policy')
        if(os.path.exists(args.base_model + 'policy')): 
            self.policy_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.target_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.transition_net.load_state_dict(th.load(args.base_model + 'transition'))
            self.policy_net.eval()
            self.target_net.eval()
            self.transition_net.eval()

        self.all_parameters = list(self.policy_net.parameters()) + list(self.transition_net.parameters())
        self.optimizer = optim.Adam(self.all_parameters, lr=args.learning_rate)
    
    def learn(self, total_timesteps):
        env = self.env
        env.reset()

        for learning_timestep in tqdm(range(total_timesteps)):
            if(learning_timestep % self.args.log_time == 0 and not self.args.compare):
                log_tag = "models-" + str(int(learning_timestep / self.args.log_time))
                self.save(self.args.save_folder, log_tag = log_tag)

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

                
    def observe(self, obs, obs_next, action, mask, mask_next, reward, done, info):
        if(obs is None): return  
        self.replay_buffer.add(obs, obs_next, action, mask, mask_next, reward, done, info) 

    def on_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        self.training_step += 1
        (obs, next_obs, acts, masks, next_masks, dones, rewards) = self.replay_buffer.sample(self.batch_size, self.env)
        
        rewards = rewards.float().squeeze() 
        masks = masks.float().squeeze()
        dones = dones.float().squeeze()
        next_masks = next_masks.float().squeeze()
        next_obs = next_obs.float().squeeze()

        acts_ohe = F.one_hot(acts, num_classes=self.action_size).squeeze()
        trans_in = th.cat((obs, acts_ohe), 1).float()
        # state, reward, done, mask 
        next_obs_pred, rewards_pred, dones_pred, next_masks_pred,  = self.transition_net(trans_in)
        rewards_pred = rewards_pred.squeeze()
        dones_pred = rewards_pred.squeeze()
        dones_pred = (dones_pred>0.9).float()           # values must be 0 or 1
        next_masks_pred = (next_masks_pred>0.9).float() # values must be 0 or 1  

        non_final_mask = th.tensor(tuple(map(lambda s: s != 1, dones_pred)), device=self.device, dtype=th.bool) 
        non_final_next_obs_pred = np.asarray([s for index, s in enumerate(next_obs_pred.cpu().detach().numpy()) if dones_pred[index] != 1])
        non_final_next_obs_pred = th.from_numpy(non_final_next_obs_pred).to(device=self.device) 

        non_final_next_masks_pred = np.asarray([s for index, s in enumerate(next_masks_pred.cpu().detach().numpy()) if dones_pred[index] != 1])
        non_final_next_masks_pred = th.from_numpy(non_final_next_masks_pred).to(device=self.device) 

        non_final_next_obs = np.asarray([s for index, s in enumerate(next_obs.cpu().detach().numpy()) if dones[index] != 1])
        non_final_next_obs = th.from_numpy(non_final_next_obs).to(device=self.device) 

        non_final_next_masks = np.asarray([s for index, s in enumerate(next_masks.cpu().detach().numpy()) if dones[index] != 1])
        non_final_next_masks = th.from_numpy(non_final_next_masks).to(device=self.device) 

        value_non_final_mask = th.tensor(tuple(map(lambda s: s != 1, dones)), device=self.device, dtype=th.bool) 

        # Compute Q(s_t, a) 
        policy_in = th.cat((obs.float(), masks.float()), 1)
        state_action_values = self.policy_net(policy_in).gather(dim=1, index=acts).squeeze()
        # Compute V(s_{t+1})
        next_state_values = th.zeros(self.batch_size, device=self.device)
        non_final_size = non_final_next_obs.size()[0]
        if( non_final_size > 0):
            # currently using true values of non final next obs and mask as input to V(s_{t+1}) 
            target_in = th.cat((non_final_next_obs.float(), non_final_next_masks.float()), 1)
            masked_next_state_values = self.target_net(target_in) 
            masked_next_state_indicies = masked_next_state_values - masked_next_state_values.min(1, keepdim=True)[0]
            masked_next_state_indicies /= masked_next_state_indicies.max(1, keepdim=True)[0] # normalize to between 0 and 1 
            masked_next_state_indicies *= non_final_next_masks
            indicies = masked_next_state_indicies.max(1)[1].detach()
            next_state_values[value_non_final_mask] = masked_next_state_values.gather(1, indicies.view(-1,1)).view(-1)

        # Compute the expected Q values
        expected_state_action_values = rewards_pred + (next_state_values * self.gamma) 

        temporal_difference = nn.SmoothL1Loss()(state_action_values, expected_state_action_values) 
        dynamics_consistency = nn.BCELoss()(next_obs_pred, next_obs) 
        reward_estimation = nn.MSELoss()(rewards_pred, rewards) 
        termination_estimation = nn.BCELoss()(dones_pred, dones)
        mask_estimation = nn.BCELoss()(next_masks_pred, next_masks) 

        loss = temporal_difference + dynamics_consistency + reward_estimation + termination_estimation + mask_estimation 
        self.loss_log.append(loss.cpu().detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.all_parameters:
            if(param.grad is None): continue # for testing 
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.training_step % self.target_update == 0: # Set target net to policy net
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
            
    def predict(self, obs, mask):
        sample = np.random.random_sample()
        self.e *= 0.99 if self.e > self.e_min else self.e_min
        if(sample < self.e):
            actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
            action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
        else:
            if(self.args.planning):
                action = best_first_search(self.policy_net, self.transition_net, obs, mask, self.args)
            else:
                with th.no_grad():
                    obs_tensor = th.from_numpy(obs).to(self.args.device)
                    mask_tensor = th.from_numpy(mask).to(self.args.device)
                    policy_in = th.cat((obs_tensor, mask_tensor))
                    q_values = self.policy_net(policy_in).cpu().numpy() 
                    masked_q_values = [q for index, q in enumerate(q_values) if mask[index] == 1]
                    action = np.where(q_values == np.max(masked_q_values))[0][0]

                    assert(False)
        
        return action

     
    def save(self, folder, log_tag):
        import os
        if not os.path.exists(folder + log_tag):
            os.makedirs(folder + log_tag)
        
        th.save(self.policy_net.state_dict(), folder + log_tag + "/policy")
        th.save(self.transition_net.state_dict(), folder + log_tag + "/transition")
    

    def compare(self, args):
        # load_folder=args.load_folder, opponent=args.opponent
        self.policy_net = Policy(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.policy_net.load_state_dict(th.load(args.load_folder + 'policy'))
        self.policy_net.eval()
        self.e = 0 

        self.transition_net = Transition(input_size=self.observation_size + self.action_size, output_size=self.observation_size + self.action_size + 2, layers=args.layers).to(self.args.device)
        self.transition_net.load_state_dict(th.load(args.load_folder + 'transition'))
        self.transition_net.eval()

        if(self.args.opponent != "random"):
            opponent_policy = Policy(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
            opponent_policy.load_state_dict(th.load(args.load_folder + 'policy'))
            opponent_policy.eval()

            opponent_transition = Transition(input_size=self.observation_size + self.action_size, output_size=self.observation_size + self.action_size + 2, layers=args.layers).to(self.args.device)
            opponent_transition.load_state_dict(th.load(args.load_folder + 'transition'))
            opponent_transition.eval()

        env = self.env
        env.reset()

        cum_rewards = []
        eps_rewards = np.zeros(len(self.env.agents))

        for _ in tqdm(range(args.timesteps)):
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
                        q_values = opponent_policy(th.from_numpy(obs).to(self.args.device)).cpu().numpy()
                        masked_q_values = -(q_values * -mask)  
                        action = np.argmax(masked_q_values)

            env.step(action) 
            rew_n, done_n = list(env.rewards.values()), list(env.dones.values())
            eps_rewards += rew_n

            if all(done_n):
                env.reset()
                cum_rewards.append(eps_rewards)
                eps_rewards = np.zeros(len(self.env.agents))
        
        return cum_rewards
