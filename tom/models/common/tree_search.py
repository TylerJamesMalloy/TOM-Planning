import torch as th 
import numpy as np


def best_first_search(policy_net, transition_net, obs, mask, args):
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
        done = trans_out[-2]
        mask = trans_out[len(obs[0]):-2]
        obs = trans_out[:obs.shape[0]]
        reward = trans_out[-1]

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