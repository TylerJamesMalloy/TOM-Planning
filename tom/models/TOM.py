import os 

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from torch import double, nn
import torch.optim as optim

from tqdm import tqdm
import copy

from argparse import ArgumentParser
from tom.models.common.networks import Policy, TOMTransition, Belief
from tom.models.common.utils import ReplayBuffer, TOMReplayBuffer
from tom.models.common.utils import get_belief_input
from tom.models.common.tree_search import best_first_search



class TOM():
    """
    Deep Model-Based Theory of Mind for planning 
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
        self.e = 1 # start at 10% exploration 
        self.training_step = 0
        self.loss_log = []

        self.prev_mask = [None for _ in env.agents]
        self.prev_obs  = [None for _ in env.agents]
        self.prev_action  = [None for _ in env.agents]
        self.prev_belief  = [[None for _ in env.agents] for _ in env.agents]
        self.prev_belief_in  = [[None for _ in env.agents] for _ in env.agents]
        self.prev_rew  = [None for _ in env.agents]
        self.prev_info  = None

        self.replay_buffer = TOMReplayBuffer(   buffer_size=int(1e6), 
                                                observation_shape=self.observation_size, 
                                                action_shape=(self.action_size,),
                                                belief_shape=(self.args.belief_in,),
                                                num_agents=len(self.agents))
         
        self.belief_net = Belief(input_size=self.args.belief_in, output_size=self.args.belief_out, layers=args.layers).to(self.args.device)
        self.policy_net = Policy(input_size=self.args.state_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.target_net = Policy(input_size=self.args.state_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)

        self.inference_net = Policy(input_size=self.args.state_size + self.num_agents, output_size=self.action_size, layers=args.layers).to(self.args.device) # add which player it is?
        self.transition_net = TOMTransition(state_size=self.args.state_size, action_size=self.action_size, mask_size=self.action_size, belief_size=self.args.belief_in, num_players=len(self.agents)-1, layers=args.layers).to(self.args.device)
        
        if(os.path.exists(args.base_model + 'policy')): 
            print(" Loading pretrained models from: ", args.base_model)
            self.policy_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.target_net.load_state_dict(th.load(args.base_model + 'policy'))
            self.transition_net.load_state_dict(th.load(args.base_model + 'transition'))
            self.belief_net.load_state_dict(th.load(args.base_model + 'belief'))
            self.policy_net.eval()
            self.target_net.eval()
            self.transition_net.eval()
            self.belief_net.eval()

        self.all_parameters = list(self.policy_net.parameters()) + list(self.transition_net.parameters()) + list(self.belief_net.parameters())
        self.optimizer = optim.Adam(self.all_parameters, lr=args.learning_rate)
    
    def learn(self, total_timesteps):
        env = self.env
        env.reset()

        all_losses = pd.DataFrame()

        for learning_timestep in tqdm(range(total_timesteps)):
            if(learning_timestep % self.args.log_time == 0 and not self.args.compare):
                log_tag = "models-" + str(int(learning_timestep / self.args.log_time))
                self.save(self.args.save_folder, log_tag = log_tag)
                #self.evaluate()

            player_index = env.agents.index(env.agent_selection)
            obs = env.observe(agent=env.agent_selection)['observation'].flatten().astype(np.float32)
            state = th.from_numpy(obs).to(self.args.device)
            info = env.unwrapped.get_perfect_information()

            belief = []
            for agent_idx, agent in enumerate(env.agents):
                if(agent != env.agent_selection):
                    belief_in = get_belief_input(env, self.prev_action[agent_idx], self.prev_belief[player_index][agent_idx])     
                    belief_in = th.from_numpy(belief_in).to(self.args.device).float()
                    belief.append(belief_in)

                    belief_out = self.belief_net(belief_in)
                    self.prev_belief[player_index][agent_idx] = belief_out
                    state = th.cat([state, belief_out])
                    

            mask = env.observe(agent=env.agent_selection)['action_mask']

            action = self.predict(state=state, mask=mask)
            env.step(action) 
            rew_n, done_n, info_n = list(env.rewards.values()), list(env.dones.values()), list(env.infos.values())
            obs_n = [env.observe(agent=agent_id)['observation'].flatten().astype(np.float32) for agent_id in env.agents]

            if(self.prev_rew[player_index] is not None and not None in self.prev_action):
                player_index_ohe = np.zeros(self.num_agents)
                player_index_ohe[player_index] = 1

                self.observe(   self.prev_obs[player_index], 
                                obs, 
                                self.prev_action, 
                                belief, 
                                self.prev_belief_in[player_index], 
                                self.prev_mask[player_index], 
                                mask, 
                                self.prev_rew[player_index], 
                                done_n[player_index], 
                                player_index_ohe,
                                info_n[player_index])
            
            self.prev_obs[player_index]    = obs
            self.prev_mask[player_index]   = mask
            self.prev_action[player_index] = action 
            self.prev_rew[player_index]    = rew_n[player_index]
            self.prev_info = info 
            self.prev_belief_in[player_index] = belief
            
            losses = self.on_step()
            if(losses is not None):
                all_losses = all_losses.append(losses, ignore_index=True)

            if all(done_n):
                if not None in self.prev_action: 
                    for agent_index in range(self.num_agents):
                        player_index_ohe = np.zeros(self.num_agents)
                        player_index_ohe[agent_index] = 1

                        mask = env.observe(agent=env.agents[agent_index])['action_mask']
                        self.observe(   self.prev_obs[agent_index], 
                                        obs_n[agent_index], 
                                        self.prev_action, 
                                        self.prev_belief_in[player_index], 
                                        self.prev_belief_in[agent_index], 
                                        self.prev_mask[agent_index], 
                                        self.prev_mask[player_index], 
                                        rew_n[agent_index], 
                                        done_n[agent_index], 
                                        player_index_ohe,
                                        info_n[agent_index])
                env.reset() 
                self.prev_mask = [None for _ in env.agents]
                self.prev_obs  = [None for _ in env.agents]
                self.prev_action  = [None for _ in env.agents]
                self.prev_belief  = [[None for _ in env.agents] for _ in env.agents]
                self.prev_belief_in  = [[None for _ in env.agents] for _ in env.agents]
                self.prev_rew  = [None for _ in env.agents]
                self.prev_info  = None
        
        return all_losses

    
    def observe(self, obs, obs_next, prev_action, belief, prev_belief, mask, mask_next, reward, done, agent, info):
        if(obs is None): return  
        """
        obs: np.ndarray,
        next_obs: np.ndarray,
        prev_action: np.ndarray,
        prev_belief: np.ndarray,
        mask:np.ndarray,
        next_mask:np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        agent: np.ndarray,
        infos: List[Dict[str, Any]]
        """
        prev_belief = th.stack([pbelief.float() for pbelief in prev_belief if pbelief is not None]).cpu().detach().numpy()
        belief = th.stack([abelief.float() for abelief in belief if abelief is not None]).cpu().detach().numpy()
        self.replay_buffer.add(obs, obs_next, prev_action, belief, prev_belief, mask, mask_next, reward, done, agent, info) 
        

    def on_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        self.training_step += 1
        (obs, next_obs, acts, beliefs, prev_beliefs, masks, next_masks, rewards, dones, agent) = self.replay_buffer.sample(self.batch_size, self.env)
        
        rewards = rewards.float().squeeze() 
        masks = masks.float().squeeze()
        dones = dones.float().squeeze()
        next_masks = next_masks.float().squeeze()
        next_obs = next_obs.float().squeeze()
        beliefs = beliefs.float().squeeze()
        prev_beliefs = prev_beliefs.float().squeeze()
        acts = acts.squeeze()
        
        idx_labels = th.from_numpy(np.argmax(agent.cpu().detach().numpy(), axis=1)).to(self.device) 
        agent_acts = th.take(acts, idx_labels)

        state = obs 
        for agent_idx in range(len(self.agents) - 1):
            if(len(self.agents) > 2):
                agent_belief = prev_beliefs[:, agent_idx, :]
            else:
                agent_belief = prev_beliefs[:, :]
            belief_out = self.belief_net(agent_belief)
            state = th.cat([state, belief_out], dim=1)

        acts_ohe = F.one_hot(agent_acts, num_classes=self.action_size).squeeze()

        trans_in = th.cat((state, acts_ohe), 1).float()
        # state, reward, done, mask 
        next_states_pred, rewards_pred, dones_pred, next_masks_pred, next_beliefs = self.transition_net(trans_in)
        rewards_pred = rewards_pred.squeeze()
        dones_pred = dones_pred.squeeze()

        for agent_idx in range(len(self.agents) - 1):
            inference_in = th.cat([next_states_pred, agent], dim=1) 
            opponent_action = self.inference_net(inference_in)

        #rand_dones =  th.rand(dones_pred.shape).to(device=self.device)
        dones_pred_sample = (dones_pred > 0.8).float()

        #rand_mask =  th.rand(next_masks_pred.shape).to(device=self.device)
        next_masks_pred_sample = (next_masks_pred > 0.8).float()
        # use actual masks and dones? 

        next_beliefs = th.reshape(next_beliefs, beliefs.shape)
        next_states = next_obs
        for agent_idx in range(len(self.agents) - 1):
            #agent_belief = beliefs[:, agent_idx, :]
            if(len(self.agents) > 2):
                agent_belief = prev_beliefs[:, agent_idx, :]
            else:
                agent_belief = prev_beliefs[:, :]
            belief_out = self.belief_net(agent_belief)
            next_states = th.cat([next_states, belief_out], dim=1)

        non_final_mask = th.tensor(tuple(map(lambda s: s != 1, dones_pred_sample)), device=self.device, dtype=th.bool) 
        non_final_next_state_pred = np.asarray([s for index, s in enumerate(next_states_pred.cpu().detach().numpy()) if dones_pred_sample[index] != 1])
        non_final_next_state_pred = th.from_numpy(non_final_next_state_pred).to(device=self.device) 

        non_final_next_masks_pred = np.asarray([s for index, s in enumerate(next_masks_pred_sample.cpu().detach().numpy()) if dones_pred_sample[index] != 1])
        non_final_next_masks_pred = th.from_numpy(non_final_next_masks_pred).to(device=self.device) 

        non_final_next_state = np.asarray([s for index, s in enumerate(next_states.cpu().detach().numpy()) if dones[index] != 1])
        non_final_next_state = th.from_numpy(non_final_next_state).to(device=self.device) 

        non_final_next_masks = np.asarray([s for index, s in enumerate(next_masks.cpu().detach().numpy()) if dones[index] != 1])
        non_final_next_masks = th.from_numpy(non_final_next_masks).to(device=self.device) 

        value_non_final_mask = th.tensor(tuple(map(lambda s: s != 1, dones)), device=self.device, dtype=th.bool) 

        # Compute Q(s_t, a) 
        policy_in = th.cat((state.float(), masks.float()), 1)
        state_action_values = self.policy_net(policy_in).gather(dim=1, index=agent_acts.unsqueeze(1)).squeeze()
        # Compute V(s_{t+1})
        next_state_values = th.zeros(self.batch_size, device=self.device)
        non_final_size = non_final_next_state.size()[0]
        if( non_final_size > 0):
            # currently using true values of non final next obs and mask as input to V(s_{t+1}) 
            target_in = th.cat((non_final_next_state.float(), non_final_next_masks.float()), 1)
            masked_next_state_values = self.target_net(target_in) 
            masked_next_state_indicies = masked_next_state_values - masked_next_state_values.min(1, keepdim=True)[0]
            masked_next_state_indicies /= masked_next_state_indicies.max(1, keepdim=True)[0] # normalize to between 0 and 1 
            masked_next_state_indicies *= non_final_next_masks
            indicies = masked_next_state_indicies.max(1)[1].detach()
            next_state_values[value_non_final_mask] = masked_next_state_values.gather(1, indicies.view(-1,1)).view(-1)

        # Compute the expected Q values
        expected_state_action_values = rewards + (next_state_values * self.gamma) 
        state_action_values =  (state_action_values * self.gamma)
        
        belief_accuracy = nn.SmoothL1Loss()(beliefs, next_beliefs)
        temporal_difference = nn.MSELoss()(state_action_values, expected_state_action_values) 
        dynamics_consistency = nn.MSELoss()(next_states_pred, next_states) 
        reward_estimation = nn.MSELoss()(rewards_pred, rewards) 
        termination_estimation = nn.SmoothL1Loss()(dones_pred, dones)
        mask_estimation = nn.SmoothL1Loss()(next_masks_pred, next_masks) 

        loss = temporal_difference + dynamics_consistency + reward_estimation + termination_estimation + mask_estimation + belief_accuracy
        self.loss_log.append(loss.cpu().detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        for param_index, param in enumerate(self.all_parameters):
            if(param.grad is None): 
                assert(False) # for testing 
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.training_step % self.target_update == 0: # Set target net to policy net
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
        
        if self.training_step % self.args.log_time == 0:

            #print(state_action_values, expected_state_action_values)
            #print(rewards, reward_estimation)
            reward_evaluation = self.evaluate()

            
            losses = {  "Temporal Loss": temporal_difference.item(), 
                        "Dynamics Loss": dynamics_consistency.item(), 
                        #"Reward Loss": reward_estimation.item(), 
                        "Termination Loss": termination_estimation.item(), 
                        "Mask Loss": mask_estimation.item(), 
                        "Belief Loss": belief_accuracy.item(),
                        "Evaluate Reward": reward_evaluation}
            print(losses)
            return  losses # only calc losses on log time steps 

        return None 
            
    def predict(self, state, mask):
        sample = np.random.random_sample()
        self.e *= 0.99 if self.e > self.e_min or self.e == 0 else self.e_min
        if(sample < self.e):
            actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
            action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
        else:
            if(self.args.planning and (not self.training_step < self.args.planning_pretraining or self.args.compare)):
                action = best_first_search(self.policy_net, self.transition_net, state, mask, self.args)
            else:
                with th.no_grad():
                    #state_tensor = th.from_numpy(state).to(self.args.device)
                    state_tensor = state 
                    mask_tensor = th.from_numpy(mask).to(self.args.device)
                    policy_in = th.cat((state_tensor, mask_tensor))
                    q_values = self.policy_net(policy_in).cpu().numpy() 
                    masked_q_values = [q for index, q in enumerate(q_values) if mask[index] == 1]
                    try:
                        action = np.where(q_values == np.max(masked_q_values))[0][0]
                    except Exception as e:
                        print(q_values)
                        print(masked_q_values)
                        actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                        action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])

        return action

    def evaluate(self):
        env = copy.copy(self.env)
        env.reset()

        original_e = copy.copy(self.e)
        self.e = 0 

        if(self.args.opponent != "random"):
            assert(False)

        env = self.env
        env.reset()

        cum_rewards = []
        eps_rewards = np.zeros(len(self.env.agents))

        prev_action  = [None for _ in env.agents]
        prev_belief  = [[None for _ in env.agents] for _ in env.agents]
        
        for _ in range(10000):
            player_key = env.agent_selection
            player_index = env.agents.index(env.agent_selection)
            obs = env.observe(agent=env.agent_selection)['observation'].flatten().astype(np.float32)
            state = th.from_numpy(obs).to(self.args.device)
            info = env.unwrapped.get_perfect_information()

            belief = []
            for agent_idx, agent in enumerate(env.agents):
                if(agent != env.agent_selection):
                    belief_in = get_belief_input(env, prev_action[agent_idx], prev_belief[player_index][agent_idx])     
                    belief_in = th.from_numpy(belief_in).to(self.args.device).float()
                    belief.append(belief_in)

                    belief_out = self.belief_net(belief_in)
                    prev_belief[player_index][agent_idx] = belief_out
                    state = th.cat([state, belief_out])
            
            mask = env.observe(agent=env.agent_selection)['action_mask']

            if(player_index == self.args.player_id):
                action = self.predict(state=state, mask=mask)
            else:
                actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
                
            prev_action[player_index] = action 

            env.step(action) 
            rew_n, done_n = list(env.rewards.values()), list(env.dones.values())
            eps_rewards += rew_n

            #print(" Player ", player_index, " with action ", action, " with mask ", mask, " and rewards: ", rew_n)

            if all(done_n):
                env.reset()
                cum_rewards.append(eps_rewards)
                eps_rewards = np.zeros(len(self.env.agents))
                prev_action  = [None for _ in env.agents]
                prev_belief  = [[None for _ in env.agents] for _ in env.agents]

        self.e = original_e
        cum_rewards = np.array(cum_rewards)
        print(" Player ", self.args.player_id, " mean reward", np.mean(cum_rewards[:,self.args.player_id]))
        #print(" Players  mean rewards", np.mean(cum_rewards, axis=0))
        return np.mean(cum_rewards[:,self.args.player_id])

    def save(self, folder, log_tag):
        import os
        if not os.path.exists(folder + log_tag):
            os.makedirs(folder + log_tag)
        
        th.save(self.policy_net.state_dict(), folder + log_tag + "/policy")
        th.save(self.transition_net.state_dict(), folder + log_tag + "/transition")
        th.save(self.belief_net.state_dict(), folder + log_tag + "/belief")
    

    def compare(self, args):
        # load_folder=args.load_folder, opponent=args.opponent
        # load from path file in seperate function 
        self.belief_net = Belief(input_size=self.args.belief_in, output_size=self.args.belief_out, layers=args.layers).to(self.args.device)
        self.policy_net = Policy(input_size=self.args.state_size + self.action_size, output_size=self.action_size, layers=args.layers).to(self.args.device)
        self.transition_net = TOMTransition(state_size=self.args.state_size, action_size=self.action_size, mask_size=self.action_size, belief_size=self.args.belief_in, num_players=len(self.agents)-1, layers=args.layers).to(self.args.device)
        
        self.policy_net.load_state_dict(th.load(args.load_folder + 'policy'))
        self.transition_net.load_state_dict(th.load(args.load_folder + 'transition'))
        self.belief_net.load_state_dict(th.load(args.load_folder + 'belief'))

        self.policy_net.eval()
        self.transition_net.eval()
        self.belief_net.eval()

        self.e = 0 

        if(self.args.opponent != "random"):
            assert(False)

        env = self.env
        env.reset()

        cum_rewards = []
        eps_rewards = np.zeros(len(self.env.agents))

        for _ in tqdm(range(args.timesteps)):
            player_key = env.agent_selection
            player_index = env.agents.index(env.agent_selection)

            player_index = env.agents.index(env.agent_selection)
            obs = env.observe(agent=env.agent_selection)['observation'].flatten().astype(np.float32)
            state = th.from_numpy(obs).to(self.args.device)
            info = env.unwrapped.get_perfect_information()

            belief = []
            for agent_idx, agent in enumerate(env.agents):
                if(agent != env.agent_selection):
                    belief_in = get_belief_input(env, self.prev_action[agent_idx], self.prev_belief[player_index][agent_idx])     
                    belief_in = th.from_numpy(belief_in).to(self.args.device).float()
                    belief.append(belief_in)

                    belief_out = self.belief_net(belief_in)
                    self.prev_belief[player_index][agent_idx] = belief_out
                    state = th.cat([state, belief_out])
            
            mask = env.observe(agent=env.agent_selection)['action_mask']

            if(player_index == args.player_id):
                action = self.predict(state=state, mask=mask)
            else:
                if(self.args.opponent == "random"):
                    actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                    action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
                else:
                    with th.no_grad():
                        assert(False)
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
