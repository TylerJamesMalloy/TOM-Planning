import os 

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from torch import nn
import torch.optim as optim

from tqdm import tqdm

from argparse import ArgumentParser
from tom.models.common.networks import Policy, Transition, Attention, Belief
from tom.models.common.utils import TOMReplayBuffer
from tom.models.common.tree_search import best_first_search

class MATOM():
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
        self.belief_size = self.observation_size * (self.num_agents ** 2)

        self.num_objects = self.env.num_objects
        self.attention_size = args.attention_size 
        
        self.model_based = args.model_based
        self.device = args.device
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.layers = args.layers
        self.model_type = args.model_type
        self.e_min = 0.05
        self.e = 0.1 # start at 10% exploration 
        self.training_step = 0
        self.loss_log = []

        self.prev_mask = [None for _ in env.agents]
        self.prev_obs  = [None for _ in env.agents]
        self.prev_beliefs = [np.zeros((self.belief_size)) for _ in env.agents]
        self.prev_action  = [None for _ in env.agents]
        self.prev_rew  = [None for _ in env.agents]
        self.prev_acts = [0 for _ in env.agents]
        

        self.replay_buffer = TOMReplayBuffer(   buffer_size=int(1e6), 
                                                observation_shape=self.observation_size, 
                                                action_shape=(self.action_size,),
                                                belief_shape=(self.belief_size),
                                                num_agents=self.num_agents)

        # belief net takes in current observation and each players previous actions, outputs full beliefs for all players
        self.belief_net = Belief(input_size= self.belief_size + (self.action_size * self.num_agents), output_size=self.belief_size, layers=args.layers).to(self.args.device)
        self.attention_net = Attention(input_size=self.observation_size * (self.num_agents ** 2), output_size=self.num_objects * self.num_agents , layers=args.layers).to(self.args.device) 
        
        self.policy_net = Policy(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.target_net = Policy(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        
        # TODO: combine the following into one net 
        self.transition_net = Transition(input_size=self.observation_size + self.action_size, output_size=self.observation_size , layers=args.layers).to(self.args.device)
        self.mask_net = Mask(input_size=self.observation_size + self.action_size, output_size= self.action_size, layers=args.layers).to(self.args.device)
        self.rewarddone_net = RewardDone(input_size=self.observation_size + self.action_size, layers=args.layers).to(self.args.device)
        self.loss_log = []

        self.nets = [self.belief_net, self.attention_net, self.policy_net, self.transition_net, self.mask_net, self.rewarddone_net]

        if(os.path.exists(args.base_model + 'policy')): 
            assert(False) # not working yet
            self.policy_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.target_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.transition_net.load_state_dict(th.load(args.base_model + 'transition'))
            self.rewarddone_net.load_state_dict(th.load(args.base_model + 'rewarddone'))
            self.mask_net.load_state_dict(th.load(args.base_model + 'mask_net'))
            self.b
            self.policy_net.eval()
            self.target_net.eval()
            self.transition_net.eval()
            self.rewarddone_net.eval()
            self.mask_net.eval()
        
        self.all_parameters = list(self.policy_net.parameters()) + list(self.transition_net.parameters()) + list(self.rewarddone_net.parameters()) + list(self.mask_net.parameters()) + list(self.belief_net.parameters()) + list(self.attention_net.parameters())
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
            prev_acts_ohe = np.array([np.eye(self.action_size)[prev_act] for prev_act in self.prev_acts ]).flatten()
            beliefs = np.concatenate((self.prev_beliefs[player_index], prev_acts_ohe))
            print(beliefs.dtype)
            beliefs = self.prev_beliefs[player_index] = self.belief_net(th.from_numpy(beliefs).to(self.args.device)).cpu().numpy()

            action = self.predict(obs=obs, mask=mask, beliefs=beliefs)
            env.step(action) 
            
            rew_n, done_n, info_n = list(env.rewards.values()), list(env.dones.values()), list(env.infos.values())
            obs_n = [env.observe(agent=agent_id)['observation'].flatten().astype(np.float32) for agent_id in self.env.agents]
            

            if(self.prev_rew[player_index] is not None):
                self.observe(self.prev_obs[player_index], obs, self.prev_acts[player_index], self.prev_mask[player_index], mask, self.prev_rew[player_index], done_n[player_index], info_n[player_index])
            
            self.prev_obs[player_index]     = obs
            self.prev_mask[player_index]    = mask
            self.prev_action[player_index]  = action 
            self.prev_rew[player_index]     = rew_n[player_index]
            self.prev_acts[player_index]    = [self.prev_action[p_i] for p_i in range(self.num_agents)] # opponent previous acts from the perspective of the current player 
            self.prev_beliefs[player_index] = beliefs

            self.on_step()

            if all(done_n):
                for agent_index in range(self.num_agents):
                    mask = env.observe(agent=env.agents[agent_index])['action_mask']
                    self.observe(self.prev_obs[agent_index], obs_n[agent_index], self.prev_acts[agent_index], self.prev_mask[agent_index], mask, rew_n[agent_index], done_n[agent_index], info_n[agent_index])
                env.reset() 
                self.prev_mask = [None for _ in env.agents]
                self.prev_obs  = [None for _ in env.agents]
                self.prev_action  = [None for _ in env.agents]
                self.prev_rew  = [None for _ in env.agents]
                self.prev_acts = [None for _ in env.agents]

                
    def observe(self, obs, obs_next, action, mask, mask_next, reward, done, info):
        if(obs is None): return  
        self.replay_buffer.add(obs, obs_next, action, mask, mask_next, reward, done, info) 

    def on_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        self.training_step += 1
        (obs, next_obs, acts, masks, next_masks, dones, rewards) = self.replay_buffer.sample(self.batch_size, self.env)
        
        print(acts.size())
        assert(False)
        rewards = rewards.float().squeeze() 
        masks = masks.float().squeeze()
        dones = dones.float().squeeze()
        next_masks = next_masks.float().squeeze()
        next_obs = next_obs.float().squeeze()

        self.belief_net()

        acts_ohe = F.one_hot(acts, num_classes=self.action_size).squeeze()
        trans_in = th.cat((obs, acts_ohe), 1).float()
        rewards_pred, dones_pred = self.rewarddone_net(trans_in)
        rewards_pred = rewards_pred.float().squeeze()
        dones_pred = dones_pred.float().squeeze()  

        next_obs_pred = self.transition_net(trans_in)
        next_masks_pred = self.mask_net(trans_in)
        #dones_pred_clip = (dones_pred>0.5).float()           # values must be 0 or 1
        #next_masks_pred_clip = (next_masks_pred>0.5).float() # values must be 0 or 1  

        non_final_mask = th.tensor(tuple(map(lambda s: s != 1, dones_pred)), device=self.device, dtype=th.bool) 
        non_final_next_obs_pred = np.asarray([s for index, s in enumerate(next_obs_pred.cpu().detach().numpy()) if dones_pred[index] != 1])
        non_final_next_obs_pred = th.from_numpy(non_final_next_obs_pred).to(device=self.device) 

        non_final_next_masks_pred = np.asarray([s for index, s in enumerate(next_masks_pred.cpu().detach().numpy()) if dones_pred[index] != 1])
        non_final_next_masks_pred = th.from_numpy(non_final_next_masks_pred).to(device=self.device) 

        # Compute Q(s_t, a) 
        state_action_values = self.policy_net(obs.float()).gather(dim=1, index=acts).squeeze()
        # Compute V(s_{t+1})
        next_state_values = th.zeros(self.batch_size, device=self.device)
        if(non_final_next_obs_pred.size()[0] > 0):
            masked_next_state_values = self.target_net(non_final_next_obs_pred.float()) * non_final_next_masks_pred
            next_state_values[non_final_mask] = masked_next_state_values.max(1).values.detach() 

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
        if(sample < self.e and not self.compare):
            actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
            action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
        else:
            if(self.args.model_based):
                action = best_first_search(self.policy_net, self.transition_net, obs, mask, self.args)
            else:
                with th.no_grad():
                    q_values = self.policy_net(th.from_numpy(obs).to(self.args.device)).cpu().numpy()
                    q_values += np.abs(np.min(q_values))
                    masked_q_values = np.zeros_like(q_values)
                    for mask_index in range(mask.shape[0]):
                        if(mask[mask_index] == 0): continue 
                        masked_q_values[mask_index] += q_values[mask_index]
                    action = np.argmax(masked_q_values)
            
        return action

     
    def save(self, folder, log_tag):
        import os
        if not os.path.exists(folder + log_tag):
            os.makedirs(folder + log_tag)
        
        th.save(self.policy_net.state_dict(), folder + log_tag + "/policy")
        th.save(self.transition_net.state_dict(), folder + log_tag + "/transition")
        th.save(self.rewarddone_net.state_dict(), folder + log_tag + "/rewarddone")
        th.save(self.mask_net.state_dict(), folder + log_tag + "/mask")
    

    def compare(self, args):
        assert(False)
        # load_folder=args.load_folder, opponent=args.opponent
        self.policy_net = MLP(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.policy_net.load_state_dict(th.load(args.load_folder + 'policy'))
        self.policy_net.eval()
        self.e = 0 

        self.transition_net = MLP(input_size=self.observation_size + self.action_size, output_size=self.observation_size + self.action_size + 2, layers=args.layers).to(self.args.device)
        self.transition_net.load_state_dict(th.load(args.load_folder + 'transition'))
        self.transition_net.eval()

        if(self.args.opponent != "random"):
            opponent = MLP(input_size=self.observation_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
            opponent.load_state_dict(th.load(args.load_folder + 'policy'))
            opponent.eval()

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
                with th.no_grad():
                    q_values = self.policy_net(th.from_numpy(obs).to(self.args.device)).cpu().numpy()
                    masked_q_values = -(q_values * -mask)  
                    action = np.argmax(masked_q_values)
            else:
                if(self.args.opponent == "random"):
                    actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                    action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
                else:
                    with th.no_grad():
                        q_values = opponent(th.from_numpy(obs).to(self.args.device)).cpu().numpy()
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


