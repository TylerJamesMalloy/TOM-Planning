import torch as th 
import numpy as np
import copy 
from torch.nn import functional as F

def get_q_values(obs, mask, policy_net, args):
    with th.no_grad():
        policy_in = th.cat((obs, mask))
        q_values = policy_net(policy_in).cpu().numpy()  
    return q_values

def best_first_search(policy_net, transition_net, obs, mask, args):
    original_mask = copy.copy(mask)
    if(not th.is_tensor(obs)):
        obs = th.from_numpy(obs).to(args.device)
    mask = th.from_numpy(mask).to(args.device)

    base_q_values = get_q_values(obs, mask, policy_net, args)
    predicted_q_values = np.zeros_like(base_q_values)
    action_size = len(predicted_q_values)
    cum_reward = 0
    current_gamma = args.gamma
    at_node = True 
    sampled_action = None
    base_obs = obs 
    depth = 0 
    max_depth = args.planning_depth * 5
    for search_depth in range(max_depth):
        if(not th.is_tensor(mask)):
            mask = th.from_numpy(np.array(mask)).to(args.device)
        q_values = get_q_values(obs, mask, policy_net, args)
        new_mask = []
        for masked_action in mask.detach().cpu().numpy() :
            if(masked_action != 0 and masked_action != 1): # skip first mask
                masked_action = np.random.choice([0, 1], p = [1-masked_action, masked_action])
            new_mask.append(masked_action)
        mask = new_mask
        if(np.sum(mask) == 0):
            done = 1 
        else:
            if(at_node): # random choice at node 
                actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
                action = sampled_action = np.random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1])
                at_node = False
            else:
                masked_q_values = [q for index, q in enumerate(q_values) if mask[index] == 1]
                action = np.where(q_values == np.max(masked_q_values))[0][0]
                
            acts_ohe = np.ones((action_size))
            acts_ohe[action] = 1
            acts_ohe = th.from_numpy(acts_ohe).to(args.device)
            trans_in = th.cat((obs, acts_ohe), 0).float()
            # state, reward, done, mask 
            obs, reward, done, mask, _ = transition_net(trans_in)
            reward = reward.cpu().detach().numpy()
            cum_reward += current_gamma * reward
            current_gamma *= current_gamma
            done_p = done.detach().cpu().numpy()[0]
            np.random.choice([0, 1], p = [1-done_p, done_p])

        if(done or search_depth == max_depth-1):
            depth += 1
            if(depth > args.planning_depth):
                break

            predicted_q_values[sampled_action] += cum_reward
            sampled_action = None
            cum_reward = 0
            obs = base_obs
            current_gamma = args.gamma
            at_node = True 
            mask = copy.copy(original_mask)
           
    
    masked_q_values = [q for index, q in enumerate(predicted_q_values) if original_mask[index] == 1]
    
    if(np.sum(masked_q_values) == 0):
        actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
        return np.random.choice([a for action_index, a in enumerate(actions) if original_mask[action_index] == 1])
    else:
        return np.where(predicted_q_values == np.max(masked_q_values))[0][0]

"""
    node_obs = obs
    q_estimate = None 
    current_q = 0
    current_gamma = args.gamma 
    samples = 0
    at_node = True 
    sampled_action = None 
    with th.no_grad(): original_q_values = -(policy_net(th.from_numpy(obs).to(args.device)).cpu().numpy() * -mask)
    q_estimate = [[] for _ in original_q_values]

    while True:
        samples += 1
        if(samples > args.planning_depth): break 
        with th.no_grad(): q_values = -(policy_net(th.from_numpy(obs).to(args.device)).cpu().numpy() * -mask)

        # Predict action based on a soft-max top k q_values 
        top_k_ind = np.argpartition(q_values, -10)[-10:]
        top_k_q = q_values[top_k_ind]
        soft_sample = np.exp(top_k_q) / np.sum(np.exp(top_k_q))
        action_index = np.random.choice(top_k_ind, 1, p=soft_sample)[0]
        action = np.zeros_like(q_values)
        action[action_index] = 1

        transition_in = np.concatenate((obs, action))
        with th.no_grad(): trans_out = transition_net(th.from_numpy(transition_in).float().to(args.device)).cpu().numpy()
        done = trans_out[-1]
        reward = trans_out[-2]
        mask = trans_out[len(obs[0]):-2]
        obs = trans_out[:obs.shape[0]]
       

        current_q += current_gamma * reward
        current_gamma *= args.gamma 

        if(at_node):
            at_node = False
            sampled_action = action_index 
        
        done = 1 if np.random.random_sample() > done else 0 # clip done to 0 or 1 
        if(done):
            q_estimate[sampled_action].append(current_q) 
            obs = node_obs
            at_node = True 
            current_gamma = args.gamma 
            current_q = 0

    q_estimate = np.asarray([0 if len(arr) == 0 else np.mean(np.asarray(arr)) for arr in q_estimate])
    action = np.argmax(q_estimate)
    return action 
"""